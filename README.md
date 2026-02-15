# Fine-Grained Segmentation for High-Resolution Scenes


This repository contains the implementation of a state-of-the-art fine-grained segmentation pipeline designed for high-resolution offroad scene understanding (544×960). This project was developed as part of the Datathon Machine Learning competition by **Team K2**.

## 🚀 Key Features

- **Native High-Resolution Processing**: Handles 544×960 images directly to preserve fine details.
- **Advanced Architecture**: Utilizes **UNet++ (Nested U-Net)** with a powerful **EfficientNet-B5** encoder.
- **Boundary Preservation**: Employs a **Combined Loss** (Cross Entropy + Dice Loss) to ensure sharp object boundaries.
- **Mixed Precision Training**: Optimized for performance using PyTorch's `autocast`.
- **Comprehensive Evaluation**: Detailed analysis including class-wise IoU, Confusion Matrices, and Boundary IoU.

## 🏗️ Architecture: UNet++ with EfficientNet-B5

The model leverages the **UNet++** architecture, which improves upon standard U-Net by:
- Using nested, dense skip connections to reduce the semantic gap between encoder and decoder features.
- Implementing deep supervision for better gradient flow and intermediate feature refinement.

The **EfficientNet-B5** backbone provides a strong balance between feature extraction capability and computational efficiency, pre-trained on ImageNet for robust low-level feature representation.

## 📊 Dataset & Classes

The project uses an offroad segmentation dataset with **10 distinct classes**:

| ID | Class Name | Description |
|---|---|---|
| 0 | Background | General environment, structures, etc. |
| 1 | Trees | Large vegetation and forestry. |
| 2 | Lush Bushes | Dense, green shrubbery. |
| 3 | Dry Grass | Yellow/brown grassy areas. |
| 4 | Dry Bushes | Non-green shrubbery or dormant bushes. |
| 5 | Ground Clutter | Small debris, leaves, and scattered items. |
| 6 | Logs | Fallen wood and timber. |
| 7 | Rocks | Stones and boulders. |
| 8 | Landscape | General terrain and earth. |
| 9 | Sky | Atmospheric area. |

## 🛠️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/annasshaikh/Datathon-MachineLearning-TeamK2.git
cd Datathon-MachineLearning-TeamK2

# Install dependencies
pip install -q segmentation-models-pytorch albumentations timm opencv-python-headless scikit-learn seaborn
```

## 📂 Project Structure

- `fine-grained-segmentation.ipynb`: Main training pipeline including data augmentation, model definition, and training loops.
- `model_evaluation_analysis.ipynb`: Comprehensive evaluation notebook for metric calculation and visualization.
- `README.md`: Project documentation.

## 📈 Performance Summary

Training results (typical):
- **Mean IoU**: ~0.XX (Update with actual final results)
- **Pixel Accuracy**: ~9X.X%
- **Best Model**: Saved as `model_best.pth` using early stopping based on Validation/Test IoU.

## 📥 Model Weights

The pre-trained model weights (`model_best.pth`) can be downloaded from the link below:

> **[Download Model Weights (Google Drive)](https://drive.google.com/file/d/1I8m9vdb-rRTquAwWCGKiH0CXzvj4uByO/view?usp=sharing)**

*(Place the downloaded `.pth` file in the `checkpoints/` directory to run evaluation)*

## 👥 Contributors (Team K2)
- **Muhammad Annas Shaikh** (Lead Developer)
- [Add other team members here]

## 📜 License
This project is for educational and competition purposes.
