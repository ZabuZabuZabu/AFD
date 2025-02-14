#!/bin/bash

VAL_DIR="/home/zabu/Deep-EIoU/sportsmot_publish/dataset/val" #path of validation data
CKPT_PATH="/home/zabu/Deep-EIoU/Deep-EIoU/checkpoints/best_ckpt.pth.tar" #path of checkpoints file
#EMB_DIR="/home/zabu/Deep-EIoU/Deep-EIoU/YOLOX_outputs/yolox_x_ch_sportsmot/emb_gt"

# 引数を解析
while getopts v:c:o: flag
do
    case "${flag}" in
        v) VAL_DIR=${OPTARG};;
        c) CKPT_PATH=${OPTARG};;
        o) OUTPUT_DIR=${OPTARG};;
        e) EMB_DIR=${OPTARG};;
    esac
done

# m_thre と for_thre の値のリスト
match_thresh_values=(0.5)
for_thresh_values=(0.9)

# 各kとcの組み合わせに対して処理を実行
for m_thre in "${match_thresh_values[@]}"; do
  for for_thre in "${for_thresh_values[@]}"; do
    echo "Running with m_thre=$m_thre and for_thre=$for_thre"

    # kとcに基づいたサブディレクトリの作成
    OUTPUT_SUBDIR="$OUTPUT_DIR/mthre_${m_thre}_fthre_${for_thre}/data"
    mkdir -p "$OUTPUT_SUBDIR"

    # Find all directories and sort them
    for VIDEO_DIR in $(find "$VAL_DIR" -mindepth 1 -maxdepth 1 -type d | sort); do
      IMG_FOLDER="$VIDEO_DIR"
      VIDEO_NAME=$(basename "$VIDEO_DIR")
      echo "Processing directory: $VIDEO_DIR"
      
      python tools/AFD_T.py --img_folder "$IMG_FOLDER" --ckpt "$CKPT_PATH" --output_dir "$OUTPUT_SUBDIR"
    done
  done
done