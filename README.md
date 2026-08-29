# AI-Based Tea Leaf Disease Classification

Final year research project — University of Kelaniya
Faculty of Computing and Technology
CSCI 43018 Research Project 2023/2024

## Overview
This research extends a ResNet50-MobileNetV2 dual-backbone 
feature fusion methodology to tea leaf disease classification 
across 8 disease categories.

## Results Summary
|     Strategy      |    Test Accuracy   |
|-------------------|--------------------|
| ResNet50 + ECA    |       95.48%       |
| MobileNetV2 + ECA |       90.83%       |
| Weighted Ensemble |       95.91%       |
| Stacking LR       |       95.98%       |
| Stacking NN       |       96.05%       |
| Feature Fusion    |       96.82%       |

## Disease Classes
- Helopeltis
- Algal leaf spot
- Red spider
- Red leaf spot
- Healthy
- Gray blight
- Brown blight
- Green mirid bug

## Project Structure
- notebooks/ — Training and evaluation notebooks (run in Colab)
- flutter_app/ — Mobile application source code
- results/ — Evaluation charts and metrics
- thesis/ — Final thesis document

## Requirements
- Python 3.10+
- PyTorch 2.x
- torchvision
- scikit-learn
- Flutter 3.x

## Supervisor
Ms. R.M.S.L. Rathnayake
Department of Software Engineering
University of Kelaniya
