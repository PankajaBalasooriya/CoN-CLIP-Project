#!/bin/bash

echo "Select the CLIP model to fine-tune:"
echo "1) ViT-B/32"
echo "2) ViT-B/16"
echo "3) ViT-L/14"
read -p "Enter your choice [1-3]: " choice

case $choice in
  1)
    clip_model="ViT-B/32"
    experiment_name="conclip_b32"
    ;;
  2)
    clip_model="ViT-B/16"
    experiment_name="conclip_b16"
    ;;
  3)
    clip_model="ViT-L/14"
    experiment_name="conclip_l14"
    ;;
  *)
    echo "Invalid choice. Exiting."
    exit 1
    ;;
esac

echo "Starting fine-tuning with model: $clip_model"

python3 conclip_fine_tuning.py \
    --clip-model-name=$clip_model \
    --experiment-name=$experiment_name \
    --negative-images=on \
    --lock-image-encoder=on \
    --batch-size=50 \
    --num-workers=4
