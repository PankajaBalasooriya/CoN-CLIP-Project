# CoN-CLIP-Project
Coursework project reimplementing CoN-CLIP with the CC-Neg dataset to study how vision-language models handle negations.

This repository contains a reimplementation of the WACV 2025 paper:

**Learning the Power of "No": Foundation Models with Negations**  
_Jaisidh Singh, Ishaan Shrivastava, Mayank Vatsa, Richa Singh, Aparna Bharati_

📄 [Paper Link](https://openaccess.thecvf.com/content/WACV2025/html/Singh_Learning_the_Power_of_No_Foundation_Models_with_Negations_WACV_2025_paper.html)
https://arxiv.org/html/2403.20312v1
---
<!--
## 🚀 Overview
Negation is a fundamental part of natural language, yet most **vision-language models (VLMs)** like CLIP fail to properly understand it.  
This repo reproduces the paper’s contributions:

1. **CC-Neg Dataset** – 228k image-caption pairs with **negated captions** for benchmarking negation understanding.
2. **CoN-CLIP Framework** – a modified contrastive learning objective that fine-tunes CLIP using negated captions and distractor images.
3. **Experiments** – evaluation of CLIP, Neg-CLIP, BLIP, FLAVA, and CoN-CLIP on:
   - Negation understanding (CC-Neg benchmark)
   - Zero-shot image classification
   - Compositionality benchmarks (SugarCREPE)

---

## 📦 Features
- Data preprocessing pipeline to generate **negated captions** (using an LLM).
- Training code for **CoN-CLIP** with modified InfoNCE loss.
- Evaluation scripts for:
  - CC-Neg benchmark
  - Zero-shot classification across 8 datasets
  - Compositional reasoning (SugarCREPE)

---

## 🛠 Installation
```bash
git clone https://github.com/<your-username>/CoN-CLIP-Reimplementation.git
cd CoN-CLIP-Reimplementation
pip install -r requirements.txt
-->
