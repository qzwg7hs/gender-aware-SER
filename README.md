# Advances in Speech Emotion Recognition Using CNNs with Feature Selection Techniques

This repository accompanies our research on enhancing Speech Emotion Recognition (SER) using a gender-aware framework that integrates Convolutional Neural Networks (CNNs) with advanced feature selection methods (Fisher Score and Feature Elimination). The study explores whether tailoring emotion classification to speaker gender improves model accuracy, generalization, and interoperability.

## Paper
- Title: Advances in Speech Emotion Recognition Using CNNs with Feature Selection Techniques
- Authors: Tayfun Kucukyilmaz, Aruay Amangeldi, Ender Sevinc, Benjamin Tyler, Ahmet Cosar, Adnan Yazici

## Project Overview
The GESCNNFS framework (Gender-Specific CNN with Feature Selection) is a fully automated pipeline that:
- Predicts speaker gender using a neural network.
- Separates training/testing data based on gender.
- Applies Fisher Score to select top features from a 193-dimensional feature set.
- Trains CNNs independently for male and female speakers.
- Evaluates model performance with both conventional and leave-one-out strategies.
- The system supports reproducible benchmarking on standard emotional speech datasets and is written in Python with TensorFlow/Keras.

## Repository Structure
```bash
├── train/
│   ├── train_gender_model.py        # Gender classification model
│   ├── train_ser_model.py           # Main SER training pipeline
│   ├── evaluate_model.py            # Evaluation scripts
├── data/
│   ├── RAVDESS.csv
│   ├── EMO-DB.csv
│   └── EMOVO.csv                    # Preprocessed features
├── utils/
│   ├── feature_selection.py         # Fisher Score + RFE
│   ├── data_split.py                # Leave-one-out and train/test
│   ├── extract_features.py          # Librosa-based feature extraction
│   └── plotting.py                  # Heatmap visualizations
├── README.md
└── requirements.txt
```

## Datasets
Three publicly available datasets were used:

| Dataset | Language | Emotions | Speakers | Samples |
|--------|-----------|----------|----------|---------|
| [RAVDESS](https://zenodo.org/record/1188976) | English | 8 (happy, sad, calm, angry, fearful, surprised, neutral, disgust) | 24 (12M/12F) | 1,440 |
| [EMO-DB](http://emodb.bilderbar.info/) | German | 7 (anger, neutral, sadness, fear, happiness, disgust, boredom) | 10 (5M/5F) | 535 |
| [EMOVO](https://github.com/fp1acm8/SER/tree/main/data/EMOVO) | Italian | 7 (disgust, fear, anger, joy, surprise, sadness, neutral) | 6 (3M/3F) | 588 |

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/qzwg7hs/gender-aware-SER.git
cd gender-aware-SER
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The main dependencies are:

- `numpy`
- `pandas`
- `librosa`
- `scikit-learn`
- `tensorflow` or `keras`
- `matplotlib`
- `seaborn`

## Usage Instructions

### 1. Feature Extraction

If you'd like to extract features from raw `.wav` files (optional, as CSVs are provided):

```bash
python utils/extract_features.py --dataset_path path_to_audio --output_csv data/output.csv
```

### 2. Train Gender Classification Model

```bash
python train/train_gender_model.py --data data/RAVDESS.csv
```

### 3. Train SER Model (Emotion Recognition)

Run the emotion classifier (gender-aware):

```bash
python train/train_ser_model.py --dataset RAVDESS --mode gender_aware
```

Run the baseline CNN without gender separation:

```bash
python train/train_ser_model.py --dataset RAVDESS --mode baseline
```

### 4. Evaluate Model Performance

```bash
python train/evaluate_model.py --dataset RAVDESS --mode gender_aware
```

## Methodology
1. Feature Extraction:
   - Extracted 193-dimensional vectors using 5 feature types: MFCC, chroma STFT, mel-spectrogram, spectral contrast, and tonnetz.
   - Used Librosa for preprocessing and feature computation.

2. Gender Classification:
   - ANN classifier using batch normalization and dropout.
   - Supports leave-one-out evaluation per speaker.

3. Feature Selection:
   - Fisher Score ranks features per class variance.
   - Top 65% features retained using Recursive Feature Elimination (RFE).

4. Emotion Recognition:
   - CNN model with 5 1D convolution layers.
   - Trained separately for male and female speakers to mitigate bias.
   - Evaluated with:
     - 80/20 train/test split.
     - Leave-One-Out Cross Validation (LOOCV).
    
## Metrics
We used the following metrics for evaluation:
- Accuracy: Percentage of correctly predicted emotion labels.
- F1-Score: Harmonic mean of precision and recall, especially useful for imbalanced classes.
- Confusion Matrices: Used for visualizing class-wise performance.

## Results Summary

We evaluated the model on three benchmark datasets under two configurations:
- **CNN with Fisher Score only** (baseline)
- **Gender-aware CNN + Fisher Score + RFE** (proposed method)

| Dataset | Configuration | Accuracy (Female) | Accuracy (Male) | Overall Accuracy |
|---------|----------------|-------------------|------------------|------------------|
| RAVDESS | CNN + Fisher   | –                 | –                | 55.21%           |
| RAVDESS | Gender + CNN + Fisher | **68.53%** | 63.89%         | 66.21%           |
| EMO-DB  | CNN + Fisher   | –                 | –                | 78.50%           |
| EMO-DB  | Gender + CNN + Fisher | **85.25%** | 80.43%         | 82.84%           |
| EMOVO   | CNN + Fisher   | –                 | –                | 75.21%           |
| EMOVO   | Gender + CNN + Fisher | **86.21%** | 84.75%         | 85.48%           |

- All evaluations use **5-fold cross-validation**.
- Gender-aware preprocessing improves accuracy consistently across all datasets.


