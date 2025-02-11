#!/bin/bash

TRAIN_DIR="/home/zabu/Deep-EIoU/sportsmot_publish/dataset/train"
CKPT_PATH="/home/zabu/Deep-EIoU/Deep-EIoU/checkpoints/best_ckpt.pth.tar"
EMB_DIR="/home/zabu/Deep-EIoU/Deep-EIoU/YOLOX_outputs/yolox_x_ch_sportsmot/emb_gt"

# 引数を解析
while getopts v:c:o: flag
do
    case "${flag}" in
        v) TRAIN_DIR=${OPTARG};;
        c) CKPT_PATH=${OPTARG};;
        o) OUTPUT_DIR=${OPTARG};;
        e) EMB_DIR=${OPTARG};;
    esac
done

#パラメータの値のリスト
mag2_value=(0.1)

# 各kとcの組み合わせに対して処理を実行
for mag2 in "${mag2_value[@]}"; do
  echo "Running with mag2=$mag2"

  # kとcに基づいたサブディレクトリの作成ｄ
  OUTPUT_SUBDIR="$OUTPUT_DIR/mag2_${mag2}/data"
  mkdir -p "$OUTPUT_SUBDIR"

  # Find all directories and sort them
  for VIDEO_DIR in $(find "$TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d | sort); do
    IMG_FOLDER="$VIDEO_DIR"
    VIDEO_NAME=$(basename "$VIDEO_DIR")
    echo "Processing directory: $VIDEO_DIR"
    
    # Run the deep_eiou script with k and c, output_dir, and association_para
    python tools/AFD_T_wgt_bbox_emb.py --img_folder "$IMG_FOLDER" --ckpt "$CKPT_PATH" --output_dir "$OUTPUT_SUBDIR" --emb_dir "$EMB_DIR" --magnification2 "$mag2" 
  done
done