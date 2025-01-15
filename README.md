# gesture_detection_emg
# Offline Machine Learning for Gesture Classification and Person Identification

## Overview
This project leverages high-density sEMG (surface electromyography) data from the CapgMyo database to develop machine learning models for:
- **Gesture Classification**
- **Person Identification**

The implementation automates data handling, preprocessing, feature extraction, and model selection to ensure optimal performance.

## Key Features
- Developed models using:
  - **Linear Discriminant Analysis (LDA)**
  - **Quadratic Discriminant Analysis (QDA)**
  - **Support Vector Machine (SVM)** with polynomial kernels.
- Automated model selection based on **F1-score** through cross-validation.
- Comprehensive Python scripts for:
  - Data preprocessing and feature extraction.
  - Training and evaluation of models.

## CapgMyo Database
The dataset considered for this project is from CapgMyo - a benchmark database for High
 Density Surface Electromyography sEMG (HD-sEMG); with recordings of gestures performed by 23
 participants (able-bodied subjects ranging in age from 23 to 26 years)
- High-density sEMG data sourced from the **CapgMyo dataset**.
- Dataset includes:
  - 18 subjects.
  - 8 gesture patterns.
  - High-resolution sEMG recordings.
  - Preprocessed data includes 1000 middle frames of data

## Results
- Achieved classification performance optimized using:
  - Cross-validation techniques.
  - Feature engineering tailored for high-density sEMG data.

## Technologies Used
- **Python**: Primary programming language.
- **Libraries**:
  - `scikit-learn`: Model training and evaluation.
  - `numpy`, `pandas`: Data preprocessing and analysis.
  - `matplotlib`: Visualization of results.

## File Structure
```plaintext
Gesture-Classification-Person-Identification/
├── data/                   # Raw and preprocessed data
├── scripts/                # Python scripts for automation
│   ├── preprocessing.py    # Data handling and preprocessing
│   ├── main.py 			 # Feature engineering, Model training, and evaluation
│   ├── Inference.py 		 # model inference
│   ├── move_files.py       # file handling
│   ├── unzip.py            # data extraction
├── model_ref/                 # Trained model files
├── results/                # Evaluation metrics and logs
└── README.md               # Project documentation
