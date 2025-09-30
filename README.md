This repository provides a complete framework for classifying medical lesions using 3D Region of Interest (ROI) data. The core of this project is a classifier built upon the Uniformer-B model, designed for high-performance video and volumetric data analysis.

Features

State-of-the-Art Model: Utilizes the powerful Uniformer-B architecture.
Cross-Validation: Implements a robust 5-fold cross-validation strategy for training and evaluation.
End-to-End Workflow: Provides clear, executable scripts for the entire pipeline: data preparation, training, inference, and model ensembling.
Reproducibility: Clear instructions to set up the environment and run the experiments.

Getting Started

Follow these instructions to set up the project on your local machine.
Prerequisites

Python 3.8+
pip for package management

Installation

Clone the repository:
bash
git clone https://github.com/puyln/OASIS.git
cd OASIS
Install the required dependencies:
bash
pip install -r requirements.txt

Usage Workflow

The project is structured into a simple, step-by-step process.
1. Data Preparation

Input Data: This model is designed to work with lesion-centered 3D ROIs. Place your dataset in the appropriate directory structure as referenced by the scripts, likely under ./data/classification_dataset.
Data Splits: We provide pre-defined data splits for 5-fold cross-validation. You can find the label files in ./data/classification_dataset/labels. You can also generate your own splits as needed.

2. Training

To train the models, you first need the base pre-trained weights for Uniformer-B.
Download Pre-trained Model:
The model is initialized with weights from the official Uniformer-B model, pre-trained on the Kinetics-400 dataset. You can find the official repository here:
Sense-X/UniFormer Official GitHub
You will need to download the specified model (Kinetics-400, #Frame:8x1x4, Sampling Stride:8), remove any layers with mismatching shapes to create a "pruned" version, and place it in the ./pretrained_weights/ directory.
Run the Training Script:
Once the pre-trained model is in place, you can start training the two sets of 5-fold cross-validation models using the following command:
bash
sh ./main/train.sh
The training progress and resulting model checkpoints will be saved.

3. Prediction

Run Inference:
Execute the prediction script to generate scores from all 10 models (2 sets of 5-fold models).
bash
sh ./main/result.sh
This will generate prediction files in the subfolders of ./results/.
Ensemble the Results:
Finally, merge the scores from all prediction files to produce a single, robust prediction.
copy
sh ./main/ensembling.sh
The final ensembled prediction file will be created at ./results/merged_score.json.
