#!/bin/bash

# SugarCREPE Evaluation Script for CLIP and CoN-CLIP

# === CONFIGURATION ===
base_dir="/home/pankaja/ENTC/Sem5/CoN-CLIP-Project"
data_root="/home/pankaja/datasets/sugarcrepe"  # Update this path
log_dir="$base_dir/logs/sugarcrepe_evaluations"
mkdir -p "$log_dir"

# Checkpoint paths
conclip_b32_ckpt="/home/pankaja/ENTC/Sem5/CoN-CLIP-Project/checkpoints/conclip_b32/ckpt_5_conclip_b32.pt"
conclip_b16_ckpt="/home/pankaja/ENTC/Sem5/CoN-CLIP-Project/checkpoints/conclip_b32/ckpt_5_conclip_b32.pt"
conclip_l14_ckpt="/home/pankaja/ENTC/Sem5/CoN-CLIP-Project/checkpoints/conclip_l14/ckpt_5_conclip_l14.pt"

echo "============================================"
echo "  SugarCREPE Evaluation for CoN-CLIP"
echo "============================================"
echo ""
echo "Select evaluation mode:"
echo "1) Single model evaluation"
echo "2) Full comparison (CLIP vs CoN-CLIP)"
read -p "Enter your choice [1-2]: " mode_choice

case $mode_choice in
  # ==============================
  # 1️⃣ SINGLE MODEL EVALUATION
  # ==============================
  1)
    echo ""
    echo "Select model to evaluate:"
    echo "1) CLIP ViT-B/32"
    echo "2) CLIP ViT-B/16"
    echo "3) CLIP ViT-L/14"
    echo "4) CoN-CLIP ViT-B/32"
    echo "5) CoN-CLIP ViT-B/16"
    echo "6) CoN-CLIP ViT-L/14"
    read -p "Enter your choice [1-6]: " model_choice

    case $model_choice in
      1)
        model="ViT-B/32"
        experiment_name="clip_b32_sugarcrepe"
        ckpt_arg=""
        ;;
      2)
        model="ViT-B/16"
        experiment_name="clip_b16_sugarcrepe"
        ckpt_arg=""
        ;;
      3)
        model="ViT-L/14"
        experiment_name="clip_l14_sugarcrepe"
        ckpt_arg=""
        ;;
      4)
        model="ViT-B/32"
        experiment_name="conclip_b32_sugarcrepe"
        if [ ! -f "$conclip_b32_ckpt" ]; then
          echo "Checkpoint not found: $conclip_b32_ckpt"
          exit 1
        fi
        ckpt_arg="--ckpt $conclip_b32_ckpt"
        ;;
      5)
        model="ViT-B/16"
        experiment_name="conclip_b16_sugarcrepe"
        if [ ! -f "$conclip_b16_ckpt" ]; then
          echo "Checkpoint not found: $conclip_b16_ckpt"
          exit 1
        fi
        ckpt_arg="--ckpt $conclip_b16_ckpt"
        ;;
      6)
        model="ViT-L/14"
        experiment_name="conclip_l14_sugarcrepe"
        if [ ! -f "$conclip_l14_ckpt" ]; then
          echo "Checkpoint not found: $conclip_l14_ckpt"
          exit 1
        fi
        ckpt_arg="--ckpt $conclip_l14_ckpt"
        ;;
      *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
    esac

    timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
    log_file="$log_dir/${experiment_name}_${timestamp}.log"
    output_file="$log_dir/${experiment_name}_${timestamp}.pt"

    echo ""
    echo "=========================================="
    echo " Model: $model"
    echo " Checkpoint: ${ckpt_arg:-None}"
    echo " Batch Size: 32"
    echo " Log: $log_file"
    echo "=========================================="
    echo ""

    python3 evaluate_sugarcrepe.py \
      --data-root "$data_root" \
      --model "$model" \
      $ckpt_arg \
      --batch-size 32 \
      --num-workers 4 \
      --device cuda \
      --output "$output_file" 2>&1 | tee "$log_file"

    echo ""
    echo "✅ Evaluation complete!"
    echo "📁 Log saved: $log_file"
    echo "📁 Results saved: $output_file"
    ;;

  # ==============================
  # 2️⃣ FULL COMPARISON
  # ==============================
  2)
    timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
    log_file="$log_dir/full_comparison_${timestamp}.log"
    output_file="$log_dir/full_comparison_${timestamp}.pt"

    echo ""
    echo "=========================================="
    echo " Running full CLIP vs CoN-CLIP comparison"
    echo " This will evaluate 6 models"
    echo " Batch Size: 32"
    echo " Log: $log_file"
    echo "=========================================="
    echo ""

    # Check if all checkpoints exist
    missing_ckpts=0
    for ckpt in "$conclip_b32_ckpt" "$conclip_b16_ckpt" "$conclip_l14_ckpt"; do
      if [ ! -f "$ckpt" ]; then
        echo "Warning: Checkpoint not found: $ckpt"
        missing_ckpts=$((missing_ckpts + 1))
      fi
    done

    if [ $missing_ckpts -gt 0 ]; then
      read -p "Some checkpoints are missing. Continue anyway? (y/n): " continue_choice
      if [[ "$continue_choice" != "y" && "$continue_choice" != "Y" ]]; then
        echo "Exiting."
        exit 1
      fi
    fi

    python3 evaluate_sugarcrepe.py \
      --data-root "$data_root" \
      --conclip-b32-ckpt "$conclip_b32_ckpt" \
      --conclip-b16-ckpt "$conclip_b16_ckpt" \
      --conclip-l14-ckpt "$conclip_l14_ckpt" \
      --batch-size 32 \
      --num-workers 4 \
      --device cuda \
      --output "$output_file" 2>&1 | tee "$log_file"

    echo ""
    echo "✅ Full comparison complete!"
    echo "📁 Log saved: $log_file"
    echo "📁 Results saved: $output_file"
    ;;

  *)
    echo "Invalid choice. Exiting."
    exit 1
    ;;
esac

echo ""
echo "Done!"