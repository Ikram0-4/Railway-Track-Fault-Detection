# Railway Track Fault Detection

This project implements an image classification system based on **CLIP (Vision Transformer)** by OpenAI.

It leverages transfer learning by using CLIP as a frozen feature extractor, and trains a downstream neural network classifier on top of the extracted visual embeddings.

---

## Features

- Pretrained CLIP (ViT-B/32) as visual backbone  
- Frozen feature extractor *(no fine-tuning of CLIP)*  
- Custom multi-layer neural classifier  
- Handling of class imbalance via weighted loss  
- Automatic threshold optimization based on F1-score  

---

## Installation

```bash
pip install -r requirements.txt
```

## Model Architecture

The model uses CLIP embeddings (768-dim) as input to a fully connected classifier:

Linear (768 → 512) + ReLU + Dropout
Linear (512 → 256) + ReLU + Dropout
Linear (256 → 2)
