# MFBF-Net

**Mamba-based Feature Bridge Fusion Network for Breast Ultrasound Lesion Segmentation**

MFBF-Net is a deep learning framework for breast ultrasound lesion segmentation, designed to effectively capture both local and global contextual information using Mamba-based feature modeling and multi-scale feature fusion.

## Overview

Breast ultrasound images often contain speckle noise, blurred lesion boundaries, and diverse lesion morphologies, making accurate lesion segmentation challenging.

MFBF-Net combines:

- Mamba-based modules for modeling long-range dependencies with efficient computational complexity.
- Multi-scale feature extraction for capturing local features at different spatial scales.
- Feature fusion to enhance information exchange between different levels of the network.
- Attention mechanisms to focus on important lesion-related features.

## Project Structure

```text
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