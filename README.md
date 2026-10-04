# Deep Learning with PyTorch 🚀

Welcome! This repository documents my journey as I foray into Deep Learning using **PyTorch**. I will be continuously updating this space with new models, experiments, and exercises as I learn and grow.

## 📌 Roadmap & Status
- [x] **Day 1: Hello World** — Pretrained Image Classification (ResNet-101)
- [ ] **Next Steps:** Custom Dataset loaders, Fine-tuning, CNNs from scratch, and NLP with PyTorch.
- *This repository is under active development. Expect regular updates!*

---

## 📂 Repository Contents

### 1. [PyTorch_QuickStart_Pretrained_models.ipynb](./PyTorch_QuickStart_Pretrained_models.ipynb)
This notebook acts as the "Hello World" entry point. Rather than training a neural network from scratch, it demonstrates the power of transfer learning by using a pre-trained **ResNet-101** network to classify an image.

#### Key Learnings:
* How to load a model pre-trained on the **ImageNet** dataset.
* Defining input preprocessing pipelines (resizing, cropping, and normalizing tensor channels).
* Performing inference and translating the output logits into human-readable class predictions.
* Evaluating model predictions and understanding background/texture biases (e.g., why a light-colored dog might get classified as an "ice bear").

---

## 🛠️ Quick Start

To run the notebooks locally, make sure you have PyTorch and Torchvision installed:

```bash
pip install torch torchvision pillow
```

Feel free to explore, open issues, or suggest improvements as I continue exploring PyTorch!
