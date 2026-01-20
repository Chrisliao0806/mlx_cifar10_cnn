# 🚀 CIFAR-10 Image Classification with MLX

<p align="center">
  <img src="https://img.shields.io/badge/Apple%20Silicon-M4%20Max-black?style=for-the-badge&logo=apple&logoColor=white" alt="Apple Silicon"/>
  <img src="https://img.shields.io/badge/MLX-Framework-orange?style=for-the-badge" alt="MLX"/>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/CIFAR--10-Dataset-green?style=for-the-badge" alt="CIFAR-10"/>
</p>

A high-performance deep learning implementation for CIFAR-10 image classification, leveraging **Apple's MLX framework** optimized for Apple Silicon. This project demonstrates a custom CNN architecture enhanced with **Self-Attention mechanisms**, achieving impressive results with native GPU acceleration on M-series chips.

---

## ✨ Highlights

- 🍎 **Native Apple Silicon Acceleration** — Built with MLX for optimal performance on M1/M2/M3/M4 chips
- 🧠 **Hybrid CNN + Self-Attention Architecture** — Combines the power of convolutional layers with attention mechanisms
- ⚡ **Blazing Fast Training** — Optimized for Apple's unified memory architecture
- 📊 **Complete Training Pipeline** — Data augmentation, cosine annealing, and comprehensive metrics tracking
- 🎯 **Production-Ready Code** — Clean, modular, and well-documented implementation

---

## 🏗️ Model Architecture

```
CifarAttentionNet
├── Prep Layer: Conv2D(3 → 64) + BatchNorm + ReLU
├── Layer 1: Conv2D(64 → 128) + MaxPool + Residual Block
├── Layer 2: Conv2D(128 → 256) + MaxPool + Self-Attention (4 heads)
├── Layer 3: Conv2D(256 → 512) + MaxPool + Residual Block
└── Classifier: Global MaxPool → Linear(512 → 10)
```

### Key Components

| Component | Description |
|-----------|-------------|
| **ConvBlock** | Conv2D → BatchNorm → ReLU with optional MaxPooling |
| **SelfAttention** | Multi-head self-attention with 4 heads for capturing global dependencies |
| **Residual Connections** | Skip connections for stable gradient flow |

---

## 🔧 Technical Specifications

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 128 |
| Epochs | 10 |
| Base Learning Rate | 0.001 |
| Optimizer | AdamW |
| Weight Decay | 1e-4 |
| LR Schedule | Cosine Annealing |

### Data Augmentation

- Random Crop (32×32 with padding=4)
- Random Horizontal Flip
- Normalization (μ=[0.4914, 0.4822, 0.4465], σ=[0.2023, 0.1994, 0.2010])

---

## 💻 Hardware & Environment

Developed and tested on:

| Spec | Details |
|------|---------|
| **Device** | MacBook Pro |
| **Chip** | Apple M4 Max |
| **RAM** | 36GB Unified Memory |
| **Framework** | MLX (Apple's ML Framework) |

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install mlx torch torchvision numpy matplotlib tqdm
```

### Run Training

Open `cifar10_mlx.ipynb` in Jupyter Notebook or VS Code and run all cells:

```bash
jupyter notebook cifar10_mlx.ipynb
```

The CIFAR-10 dataset will be automatically downloaded on first run.

---

## 📁 Project Structure

```
mlx_cifar10_cnn/
├── cifar10_mlx.ipynb    # Main training notebook
├── data/                 # CIFAR-10 dataset (auto-downloaded)
│   └── cifar-10-batches-py/
├── README.md
└── LICENSE
```

---

## 📈 Training Features

- **Real-time Progress Tracking** — tqdm progress bars with live loss/accuracy updates
- **Learning Rate Visualization** — Dynamic LR shown during training
- **History Plotting** — Automatic generation of loss/accuracy curves
- **Validation Metrics** — Comprehensive evaluation on test set after each epoch

---

## 🎯 Why MLX?

[MLX](https://github.com/ml-explore/mlx) is Apple's machine learning framework designed specifically for Apple Silicon. Key advantages:

1. **Unified Memory** — No CPU↔GPU data transfer overhead
2. **Lazy Evaluation** — Efficient computation graph execution
3. **NumPy-like API** — Familiar and intuitive syntax
4. **Native Performance** — Optimized for M-series Neural Engine and GPU

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Apple MLX Team](https://github.com/ml-explore/mlx) for the amazing framework
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) by Alex Krizhevsky
- PyTorch team for data loading utilities

---

<p align="center">
  <b>Built with ❤️ on Apple Silicon</b>
</p>