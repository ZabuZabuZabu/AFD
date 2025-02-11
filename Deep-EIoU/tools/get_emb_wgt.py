import argparse
import os
import os.path as osp
import numpy as np
import time
import cv2
import torch
import sys
sys.path.append('.')

import gc
from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracking_utils.timer import Timer

from tracker.AFD import AFD
from reid.torchreid.utils import FeatureExtractor
import torchvision.transforms as T


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def make_parser():
    parser = argparse.ArgumentParser("DeepEIoU Demo")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="../demo.mp4", help="path to images or video"
    )
    parser.add_argument(
        "--save_result",
        default=True,
        help="whether to save the inference result of image/video",
    )

    # 追加: 画像フォルダのパス
    parser.add_argument(
        "--img_folder", default=None, type=str, help="path to image folder"
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default="yolox/yolox_x_ch_sportsmot.py",
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold valid for tracks")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=60, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--nms_thres", type=float, default=0.7, help='nms threshold')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    # reid args
    parser.add_argument("--with-reid", dest="with_reid", default=True, action="store_true", help="use Re-ID flag.")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')

    parser.add_argument("--output_dir", type=str, default=None, help="directory to save the output results")  # 追加
    parser.add_argument("--association_para", type=float, default=0.9, help="high parameter for association")  # 追加
    parser.add_argument("--pos_thresh", type=float, default=0.08, help="threshold for position vector distance") #追加
    parser.add_argument("--edge_margin", type=float, default=20, help="margin size for eliminating bbox in edge") #追加
    parser.add_argument("--magnification", type=float, default=20, help="magnification for GED") #追加
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        if osp.basename(maindir) == 'img1':  # 'img1' ディレクトリのみを対象とする
            for filename in file_name_list:
                apath = osp.join(maindir, filename)
                ext = osp.splitext(apath)[1]
                if ext in IMAGE_EXT:
                    image_names.append(apath)
    return image_names



def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        return outputs, img_info

#gt.txtからBBOXの座標とスコアだけ持ってくる
def load_gt(gt_file):
    gt_dict = {}
    with open(gt_file, 'r') as f:
        for line in f:
            frame_id, track_id, x1, y1, w, h, _, _, _ = map(float, line.strip().split(','))
            if int(frame_id) not in gt_dict:
                gt_dict[int(frame_id)] = []
            gt_dict[int(frame_id)].append([x1, y1, x1 + w, y1 + h, 1.0, 1.0, 0])
    return gt_dict

def imageflow_demo(predictor, extractor, output_video_dir, current_time, args):
    #データセットからgt持ってくる
    img_folder = args.img_folder
    gt_file_path = os.path.join(img_folder, 'gt', 'gt.txt')
    gt_data = load_gt(gt_file_path)  # GTファイルのパス

    if args.img_folder:
        img_list = sorted(get_image_list(args.img_folder))
        video_name = osp.basename(args.img_folder)
        width = cv2.imread(img_list[0]).shape[1]
        height = cv2.imread(img_list[0]).shape[0]
        fps = args.fps
    else:
        cap = cv2.VideoCapture(args.path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_name = osp.splitext(osp.basename(args.path))[0]

    tracker = AFD(args, frame_rate=30)
    timer = Timer()
    frame_id = 1
    results = []
    # 特徴量保存用リストを初期化
    all_embeddings = []  # 各フレームの特徴量を保存するリスト

    while True:
        if frame_id % 30 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        if args.img_folder:
            if frame_id > len(img_list):
                break
            frame = cv2.imread(img_list[frame_id - 1])
        else:
            ret_val, frame = cap.read()
            if not ret_val:
                break

        # GTデータを使用
        if frame_id in gt_data:
            det = np.array(gt_data[frame_id])

            # 以下、推論部分をGTデータに基づいて処理
            valid_crops = []
            valid_dets = []
            for i, (x1, y1, x2, y2, _, _, _) in enumerate(det):
                x1, y1, x2, y2 = map(int, [max(0, x1), max(0, y1), min(width, x2), min(height, y2)])
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        valid_crops.append(crop)
                        valid_dets.append(det[i])

            if not valid_crops:
                logger.warning(f"No valid crops in frame {frame_id}")
                all_embeddings.append([])  # 特徴量なしの場合は空リストを保存

            try:
                embs = extractor(valid_crops)
                embs = embs.cpu().detach().numpy()
                det = np.array(valid_dets)
                all_embeddings.append(embs)  # フレーム単位で特徴量を保存
            except Exception as e:
                logger.error(f"Error in frame {frame_id}: {e}")
                all_embeddings.append([])  # エラー時も空リストを保存
                continue

        frame_id += 1

    # 特徴量をファイルに保存
    if args.save_result:
        save_file = os.path.join(output_video_dir, f'{video_name}.npy')
        np.save(save_file, np.array(all_embeddings, dtype=object))  # 2DリストをNumpy形式で保存
        logger.info(f"Saved GT features to {save_file}")


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    vis_folder = osp.join(output_dir, "track_vis")
    os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = "checkpoints/best_ckpt.pth.tar"
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()

    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='checkpoints/sports_model.pth.tar-60',
        device='cuda'
    )

    imageflow_demo(predictor, extractor, args.output_dir, current_time, args)



if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)

