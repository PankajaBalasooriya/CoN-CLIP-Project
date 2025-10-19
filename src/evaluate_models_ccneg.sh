#!/bin/bash

# === CONFIGURATION ===
base_dir="/home/pankaja/ENTC/Sem5/CoN-CLIP-Project"
log_dir="$base_dir/logs/evaluations"
mkdir -p "$log_dir"

# === MAIN MENU ===
echo "Select the model family to evaluate:"
echo "1) CLIP (OpenAI)"
echo "2) CoN-CLIP (Fine-tuned)"
echo "3) FLAVA"
echo "4) BLIP"
read -p "Enter your choice [1-4]: " family_choice

case $family_choice in
  # ==============================
  # 1️⃣ CLIP FAMILY
  # ==============================
  1)
    echo "Select the CLIP model architecture:"
    echo "1) ViT-B/32"
    echo "2) ViT-B/16"
    echo "3) ViT-L/14"
    read -p "Enter your choice [1-3]: " clip_choice

    case $clip_choice in
      1) model="ViT-B/32"; experiment_name="clip_b32";;
      2) model="ViT-B/16"; experiment_name="clip_b16";;
      3) model="ViT-L/14"; experiment_name="clip_l14";;
      *) echo "Invalid choice. Exiting."; exit 1;;
    esac

    ckpt_arg=""
    eval_script="evaluate_ccneg.py"
    eval_args="--model \"$model\" $ckpt_arg --device cuda --batch 50 --workers 4"
    ;;

  # ==============================
  # 2️⃣ CoN-CLIP FAMILY
  # ==============================
  2)
    echo "Select the CoN-CLIP model architecture:"
    echo "1) ViT-B/32"
    echo "2) ViT-B/16"
    echo "3) ViT-L/14"
    read -p "Enter your choice [1-3]: " conclip_choice

    case $conclip_choice in
      1)
        model="ViT-B/32"
        experiment_name="conclip_b32"
        ckpt_path="$base_dir/logs/conclip_b32/results_conclip_b32.pt"
        ;;
      2)
        model="ViT-B/16"
        experiment_name="conclip_b16"
        ckpt_path="$base_dir/logs/conclip_b16/results_conclip_b16.pt"
        ;;
      3)
        model="ViT-L/14"
        experiment_name="conclip_l14"
        ckpt_path="$base_dir/logs/conclip_l14/results_conclip_l14.pt"
        ;;
      *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
    esac

    if [ ! -f "$ckpt_path" ]; then
      echo "Checkpoint not found at: $ckpt_path"
      echo "Exiting."
      exit 1
    fi
    ckpt_arg="--ckpt $ckpt_path"
    eval_script="evaluate_ccneg.py"
    eval_args="--model \"$model\" $ckpt_arg --device cuda --batch 50 --workers 4"
    ;;

  # ==============================
  # 3️⃣ FLAVA MODEL
  # ==============================
  3)
    model="flava"
    experiment_name="flava_eval"
    eval_script="evaluate_flava.py"
    eval_args="--device cuda --batch 8"
    ckpt_arg=""
    ;;

  # ==============================
  # 4️⃣ BLIP MODEL
  # ==============================
  4)
    model="blip"
    experiment_name="blip_eval"
    eval_script="evaluate_blip.py"
    eval_args="--device cuda --batch 8"
    ckpt_arg=""
    ;;

  *)
    echo "Invalid selection. Exiting."
    exit 1
    ;;
esac

# === LOGGING ===
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
log_file="$log_dir/${experiment_name}_${timestamp}.log"

echo "----------------------------------------"
echo " Evaluating Model: $model"
if [[ -n "$ckpt_arg" ]]; then
  echo " Checkpoint: $ckpt_path"
else
  echo " Checkpoint: None"
fi
echo " Device: cuda (GPU)"
echo " Log File: $log_file"
echo "----------------------------------------"

# === RUN EVALUATION ===
echo "Running: python3 $eval_script $eval_args"
eval python3 $eval_script $eval_args 2>&1 | tee "$log_file"

echo "----------------------------------------"
echo "✅ Evaluation complete for $model"
echo "📁 Results saved in: $log_file"
echo "----------------------------------------"
