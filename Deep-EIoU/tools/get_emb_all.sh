#!/bin/bash
CKPT_PATH="/home/zabu/Deep-EIoU/Deep-EIoU/checkpoints/best_ckpt.pth.tar"
VAL_DIR="/home/zabu/Deep-EIoU/sportsmot_publish/dataset/val"

# 引数を解析
while getopts v:c:o:a: flag
do
    case "${flag}" in
        v) VAL_DIR=${OPTARG};;
        c) CKPT_PATH=${OPTARG};;
        o) OUTPUT_DIR=${OPTARG};;
    esac
done

# 引数が指定されていない場合のデフォルト値
# OUTPUT_DIR=${OUTPUT_DIR:-"/path/to/output"}

# Find all directories and sort them
for VIDEO_DIR in $(find "$VAL_DIR" -mindepth 1 -maxdepth 1 -type d | sort); do
  IMG_FOLDER="$VIDEO_DIR"
  VIDEO_NAME=$(basename "$VIDEO_DIR")
  echo "Processing directory: $VIDEO_DIR"

  # Create output directory for each video
  #OUTPUT_VIDEO_DIR="$OUTPUT_DIR"
  #mkdir -p "$OUTPUT_VIDEO_DIR"

  # Run the deep_eiou script with output_dir and association_para
  python tools/get_emb_wgt.py --img_folder "$IMG_FOLDER" --ckpt "$CKPT_PATH" --output_dir "$OUTPUT_DIR" 
done
