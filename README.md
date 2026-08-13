MFBF-Net

Mamba-based Feature Bridge Fusion Network for Breast Ultrasound Lesion Segmentation

MFBF-Net is a deep learning framework for breast ultrasound lesion segmentation, designed to effectively capture both local and global contextual information using Mamba-based feature modeling and multi-scale feature fusion.

Overview

Breast ultrasound images often contain speckle noise, blurred lesion boundaries, and diverse lesion morphologies, making accurate lesion segmentation challenging.

MFBF-Net combines:

Mamba-based modules for modeling long-range dependencies with efficient computational complexity.
Multi-scale feature extraction for capturing local features at different spatial scales.
Feature fusion to enhance information exchange between different levels of the network.
Attention mechanisms to focus on important lesion-related features.
Project Structure
MFBF-Net/
├── model/
│   ├── DMBF.py
│   ├── FMPA.py
│   ├── MFBF_Net.py
│   ├── PMAA.py
│   ├── VSS.py
│   └── __init__.py
│
├── DataLoader.py
├── Metrics.py
├── Train.py
├── Test.py
├── MFBF_Net.ipynb
├── requirements.txt
└── README.md
Requirements

Install the required Python packages:

pip install -r requirements.txt

The main dependencies include:

PyTorch
TorchVision
PyTorch Lightning
NumPy
OpenCV
Pillow
Matplotlib
Scikit-learn
Dataset

This project is developed for breast ultrasound lesion segmentation.

The training and testing pipeline expects the dataset to provide:

images
masks

The dataset is loaded using the BUSILoader class implemented in DataLoader.py.

Data Augmentation

The training pipeline supports several augmentation techniques, including:

Random resized crop
Random rotation
Horizontal flip
Vertical flip
Elastic deformation
Brightness and contrast adjustment
Speckle noise augmentation
Training

After preparing the dataset and configuring the data path in Train.py, run:

python Train.py

The training pipeline uses:

Optimizer: Adam
Initial learning rate: 1e-3
Batch size: 8
Loss: Dice-Tversky loss
Evaluation metrics: Dice and IoU
Testing

Configure the dataset path and model checkpoint path in Test.py, then run:

python Test.py

The model is evaluated using:

Dice Similarity Coefficient (DSC)
Intersection over Union (IoU)
Evaluation Metrics
Dice Similarity Coefficient

Dice measures the overlap between the predicted segmentation and the ground-truth mask.

Intersection over Union

IoU measures the ratio between the intersection and union of the predicted and ground-truth regions.

Model

The main model is implemented in:

model/MFBF_Net.py

The model directory contains the main components used by MFBF-Net:

DMBF.py
FMPA.py
PMAA.py
VSS.py
MFBF_Net.py