#!/bin/bash

# Create log directory if not exists
log_dir="/home/pankaja/ENTC/Sem5/CoN-CLIP-Project/logs/evaluations"
mkdir -p "$log_dir"

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

# Ask whether to use the fine-tuned CoN-CLIP model
read -p "Do you want to evaluate the fine-tuned CoN-CLIP model for $clip_model? (y/n): " use_conclip

if [[ "$use_conclip" == "y" || "$use_conclip" == "Y" ]]; then
  if [ ! -f "$ckpt_path" ]; then
    echo "Checkpoint file not found at: $ckpt_path"
    echo "Exiting."
    exit 1
  fi
  ckpt_arg="--ckpt $ckpt_path"
  echo "Using fine-tuned CoN-CLIP checkpoint for $clip_model."
else
  ckpt_arg=""
  echo "Using base CLIP model (no fine-tuning)."
fi

# Generate timestamped log file
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
log_file="$log_dir/${experiment_name}_${timestamp}.log"

echo "----------------------------------------"
echo " Evaluating Model: $clip_model"
echo " Checkpoint: ${ckpt_path:-None}"
echo " Device: cuda (GPU)"
echo " Batch Size: 50"
echo " Workers: 4"
echo " Log File: $log_file"
echo "----------------------------------------"

# Run evaluation and tee output to both terminal and log file
python3 evaluate_ccneg.py \
  --model "$clip_model" \
  $ckpt_arg \
  --device cuda \
  --batch 50 \
  --workers 4 2>&1 | tee "$log_file"

echo "----------------------------------------"
echo "Evaluation complete for $clip_model"
echo "Results saved in: $log_file"
echo "----------------------------------------"
