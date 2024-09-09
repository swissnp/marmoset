# Marmoset Pose Estimation Repository

This repository contains tools and models for performing pose estimation on marmoset monkeys using deep learning techniques. The project is divided into two main parts, each handled by a separate Jupyter notebook:

### 1. `YOLO.ipynb`

This notebook details the use of the YOLO (You Only Look Once) model adapted for pose estimation. It describes how the YOLO model is configured to detect keypoints of marmoset monkeys in image data. The process includes:

- Preparing and loading the dataset.
- Training the YOLO model with pose estimation capabilities.
- Evaluating the model performance on validation data.
- Saving the best-performing model weights.

### 2. `SAM_joint_estimation.ipynb`

This notebook focuses on the Semantic Attention Model (SAM) for joint estimation. SAM is used after initial keypoint detection by YOLO to refine the estimation of marmoset joints in images. The notebook covers:

- Setting up the SAM model with pre-trained weights.
- Processing images through the SAM to enhance the accuracy of joint localization.
- Visualizing the output with masks and skeleton overlays on the images.

### Pretrained Model: `marmo.pt`

The repository also includes `marmo.pt`, a pretrained model that represents the trained YOLO model with optimized weights for marmoset pose estimation. This model can be directly used for inference or further fine-tuning on similar datasets.

## Repository Structure

```
marmo-4b/
│
├── YOLO.ipynb              - Notebook for training and evaluating the YOLO model.
├── SAM_joint_estimation.ipynb  - Notebook for detailed joint estimation using SAM.
└── marmo.pt                - Pretrained model weights.
```

## High-Level Workflow

1. **Dataset Preparation**: Organize and preprocess images and annotations for training and validation.
2. **Model Training**: Utilize `YOLO.ipynb` to train the YOLO model on the marmoset dataset to detect initial keypoints.
3. **Joint Estimation**: Apply the `SAM_joint_estimation.ipynb` to refine keypoints and estimate joint positions more accurately using the SAM model.
4. **Model Inference**: Use the pretrained `marmo.pt` for making predictions on new data or further refine the model based on additional marmoset images.

## Usage

To replicate the results or to train the models on your data, follow the setup and execution steps described in each notebook. Ensure all dependencies are installed as per the notebooks' instructions.

## Contributions

Contributions to this project are welcome. You can improve the existing model, extend the dataset, or enhance the notebooks by following the contribution guidelines laid out in this repository.
