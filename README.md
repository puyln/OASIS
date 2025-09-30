# OASIS: A 3D Medical Lesion Classification Framework based on Uniformer-B

This repository provides a complete, end-to-end framework for classifying medical lesions using 3D Region of Interest (ROI) data. The core of this project is a classifier built upon the **Uniformer-B** model, an advanced architecture designed for high-performance video and volumetric data analysis.

## Project Overview

In the field of medical image analysis, accurately classifying 3D lesions from scans like CT or MRI is crucial for assisting clinical diagnosis. This project aims to provide a reproducible, high-performance solution that automatically learns discriminative features from 3D ROIs to deliver precise classification predictions.

The framework covers the entire pipeline, from data preparation, model training, and cross-validation to inference and final result ensembling, providing researchers and developers with a solid foundation to build upon.

## Model Architecture: Uniformer-B

The classifier's backbone is the **Uniformer** model, an innovative architecture that merges the strengths of **Convolutional Neural Networks (CNNs)** and **Transformers**.

-   **Local Feature Extraction**: Unlike standard Vision Transformers, Uniformer employs convolutions in its early layers to efficiently capture local textures and spatial details, which is vital for identifying the subtle characteristics of medical lesions.
-   **Global Dependency Modeling**: In its deeper layers, the model leverages the Transformer's self-attention mechanism to capture long-range dependencies within the 3D volume, enabling it to understand the lesion's overall structure and context.
-   **Efficiency and Performance**: This hybrid design allows Uniformer to achieve powerful performance while maintaining computational efficiency, making it well-suited for processing resource-intensive 3D volumetric data.

We utilize the **Uniformer-B** model, pre-trained on the **Kinetics-400** dataset, and adapt it to the medical lesion classification task via transfer learning.

## ✨ Features

-   **State-of-the-Art Model**: Utilizes the powerful Uniformer-B architecture, designed for temporal and volumetric data.
-   **Cross-Validation**: Implements a robust 5-fold cross-validation strategy for training and evaluation to ensure model generalization.
-   **End-to-End Workflow**: Provides clear, executable scripts for the entire pipeline: data preparation, training, inference, and model ensembling.
-   **Reproducibility**: Includes clear instructions to set up the environment and run the experiments.

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

-   Python 3.8+
-   PyTorch
-   `pip` for package management

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/puyln/OASIS.git
    cd OASIS
    ```

2.  **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Usage Workflow

The project is structured into a simple, three-step process.

### 1. Data Preparation

-   **Input Data**: This model is designed to work with lesion-centered 3D ROIs. Place your dataset in the appropriate directory structure as referenced by the scripts, likely under `./data/classification_dataset`.
-   **Data Splits**: We provide pre-defined data splits for 5-fold cross-validation. You can find the label files in `./data/classification_dataset/labels`. You can also generate your own splits as needed.

### 2. Model Training

To train the models, you first need the base pre-trained weights for Uniformer-B.

-   **Download Pre-trained Model**:
    1.  The model is initialized with weights from the official Uniformer-B model, pre-trained on the **Kinetics-400** dataset. You can find the official repository here:
        [Sense-X/UniFormer Official GitHub](https://github.com/Sense-X/UniFormer)
    2.  You will need to download the specified model version (**Kinetics-400, #Frame:8x1x4, Sampling Stride:8**).
    3.  Since the downstream task has a different classification head, you must remove any layers with mismatching shapes from the weight file (e.g., the final classification layer) to create a "pruned" version.
    4.  Place the processed weight file in the `./pretrained_weights/` directory.

-   **Run the Training Script**:
    Once the pre-trained model is in place, you can start training the two sets of 5-fold cross-validation models (10 models in total) using the following command:
    ```bash
    sh ./main/train.sh
    ```
    The training progress and resulting model checkpoints will be saved in their respective directories.

### 3. Prediction and Ensembling

-   **Run Inference**:
    Execute the prediction script to generate scores from all 10 trained models.
    ```bash
    sh ./main/result.sh
    ```
    This will generate prediction files in the subfolders of `./results/`.

-   **Ensemble the Results**:
    Finally, merge the scores from all prediction files to produce a single, more robust prediction.
    ```bash
    sh ./main/ensembling.sh
    ```
    The final ensembled prediction file will be created at `./results/merged_score.json`.
