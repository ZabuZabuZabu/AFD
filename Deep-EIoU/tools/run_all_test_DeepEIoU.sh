#!/bin/bash

TEST_DIR="/home/zabu/Deep-EIoU/sportsmot_publish/dataset/test"
CKPT_PATH="/home/zabu/Deep-EIoU/Deep-EIoU/checkpoints/best_ckpt.pth.tar"
OUTPUT_DIR="/home/zabu/Deep-EIoU/Deep-EIoU/YOLOX_outputs/yolox_x_ch_sportsmot/track_vis/DeepEIoU_test2"

# Find all directories and sort them
for VIDEO_DIR in $(find "$TEST_DIR" -mindepth 1 -maxdepth 1 -type d | sort); do
  IMG_FOLDER="$VIDEO_DIR"
  VIDEO_NAME=$(basename "$VIDEO_DIR")
  echo "Processing directory: $VIDEO_DIR"

  # # Create output directory for each video
  # OUTPUT_VIDEO_DIR="$OUTPUT_DIR/$VIDEO_NAME"
  # mkdir -p "$OUTPUT_VIDEO_DIR"

  # Run the deep_eiou script
  python tools/DeepEIoU_T.py --img_folder "$IMG_FOLDER" --ckpt "$CKPT_PATH" --output_dir "$OUTPUT_DIR"
done
