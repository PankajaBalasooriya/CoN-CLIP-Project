#!/bin/bash

echo "Select the CLIP model to evaluate:"
echo "1) ViT-B/32"
echo "2) ViT-B/16"
echo "3) ViT-L/14"
read -p "Enter your choice [1-3]: " choice

case $choice in
  1)
    clip_model="ViT-B/32"
    experiment_name="eval_b32"
    ckpt_path="/home/pankaja/ENTC/Sem5/CoN-CLIP-Project/logs/conclip_b32/results_conclip_b32.pt"
    ;;
  2)
    clip_model="ViT-B/16"
    experiment_name="eval_b16"
    ckpt_path="/home/pankaja/ENTC/Sem5/CoN-CLIP-Project/logs/conclip_b16/results_conclip_b16.pt"
    ;;
  3)
    clip_model="ViT-L/14"
    experiment_name="eval_l14"
    ckpt_path="/home/pankaja/ENTC/Sem5/CoN-CLIP-Project/logs/conclip_l14/results_conclip_l14.pt"
    ;;
  *)
    echo "Invalid choice. Exiting."
    exit 1
    ;;
esac

# Check if the checkpoint file exists
if [ ! -f "$ckpt_path" ]; then
  echo "⚠️  Warning: Checkpoint not found at $ckpt_path"
  read -p "Do you still want to continue without checkpoint? (y/n): " cont
  if [[ "$cont" != "y" && "$cont" != "Y" ]]; then
    echo "Exiting."
    exit 1
  else
    ckpt_arg=""
  fi
else
  ckpt_arg="--ckpt $ckpt_path"
fi

echo "----------------------------------------"
echo " Evaluating Model: $clip_model"
echo " Checkpoint: ${ckpt_path:-None}"
echo " Device: cuda (GPU)"
echo " Batch Size: 50"
echo " Workers: 4"
echo "----------------------------------------"

python3 evaluate_ccneg.py \
  --model "$clip_model" \
  $ckpt_arg \
  --device cuda \
  --batch 50 \
  --workers 4

echo "----------------------------------------"
echo " Evaluation complete for $clip_model"
echo "----------------------------------------"
