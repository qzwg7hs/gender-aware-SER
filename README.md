# Advances in Speech Emotion Recognition1 Using a CNN and Gender-Based Segmentation framework with Feature Selection Techniques

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
├── data/                                  # Preprocessed features
│   ├── combined_speech_data.csv
│   ├── featureEMOVO.csv
│   ├── featureEmodb_Ek.csv
│   └── featureNormal_Ek.csv               
├── train/                          
│   ├── EMODB/
│   │   ├── EmoDB_gender_both.ipynb        # 80-20 train/test split
│   │   ├── EmoDB_gender_female.ipynb           
│   │   ├── EmoDB_gender_male.ipynb            
│   │   ├── EmoDB_speaker_both.ipynb       # Speaker independent leave-one-out split
│   │   ├── EmoDB_speaker_female.ipynb
│   │   ├── EmoDB_speaker_male.ipynb
│   ├── EMOVO/
│   │   ├── EMOVO_gender_both.ipynb      
│   │   ├── EMOVO_gender_female.ipynb    
│   │   ├── EMOVO_gender_male.ipynb      
│   │   ├── EMOVO_speaker_both.ipynb
│   │   ├── EMOVO_speaker_female.ipynb
│   │   ├── EMOVO_speaker_male.ipynb
│   ├── RAVDESS/
│   │   ├── ravdess_gender_both.ipynb        
│   │   ├── ravdess_gender_female.ipynb           
│   │   ├── ravdess_gender_male.ipynb           
│   │   ├── ravdess_speaker_both.ipynb
│   │   ├── ravdess_speaker_female.ipynb
│   │   ├── ravdess_speaker_male.ipynb
│   ├── gender.py                           # Gender classification model
├── utils/
│   ├── EMODB_Writecsv.ipynb                # Feature extraction
│   ├── EMOVO_Writecsv.ipynb
│   ├── RAVDESS_Writecsv.ipynb
│   ├── combine_csv.py                      # Helper function
│   ├── diagram.py                          # Helper function for .wav visualization
├── README.md
└── requirements.txt
```

## Datasets
Three publicly available datasets were used:

| Dataset | Language | Emotions | Speakers | Samples |
|--------|-----------|----------|----------|---------|
| [RAVDESS](https://zenodo.org/record/1188976) | English | 8 (happy, sad, calm, angry, fearful, surprised, neutral, disgust) | 24 (12M/12F) | 1,440 |
| [EMO-DB](http://emodb.bilderbar.info/index-1280.html) | German | 7 (anger, boredom, anxiety, happiness, sadness, disgust, neutral) | 10 (5M/5F) | 535 |
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

### 1. Feature Extraction (Optional)

Preprocessed CSV feature files are already included in the `data/` directory.  
If you wish to regenerate them from raw `.wav` files:

- For EMO-DB:
  Open and run:
  ```
  utils/EMODB_Writecsv.ipynb
  ```

- For EMOVO:
  Open and run:
  ```
  utils/EMOVO_Writecsv.ipynb
  ```

- For RAVDESS:
  Open and run:
  ```
  utils/RAVDESS_Writecsv.ipynb
  ```

---

### 2. Train Gender Classification Model

```bash
python train/gender.py
```
This model predicts speaker gender from audio features and is used to route samples into gender-specific SER pipelines.

---

### 3. Train & Evaluate Emotion Recognition Models

All model training and evaluation is done via Jupyter Notebooks organized by dataset and split strategy.

| Dataset | 80–20 Split                      | Leave-One-Out (Speaker Independent)      |
|---------|----------------------------------|------------------------------------------|
| EMO-DB  | `train/EMODB/EmoDB_gender_both.ipynb`<br>`EmoDB_gender_female.ipynb`<br>`EmoDB_gender_male.ipynb` | `EmoDB_speaker_both.ipynb`<br>`EmoDB_speaker_female.ipynb`<br>`EmoDB_speaker_male.ipynb` |
| EMOVO   | `train/EMOVO/EMOVO_gender_both.ipynb`<br>`EMOVO_gender_female.ipynb`<br>`EMOVO_gender_male.ipynb` | `EMOVO_speaker_both.ipynb`<br>`EMOVO_speaker_female.ipynb`<br>`EMOVO_speaker_male.ipynb` |
| RAVDESS | `train/RAVDESS/ravdess_gender_both.ipynb`<br>`ravdess_gender_female.ipynb`<br>`ravdess_gender_male.ipynb` | `ravdess_speaker_both.ipynb`<br>`ravdess_speaker_female.ipynb`<br>`ravdess_speaker_male.ipynb` |

Each notebook:
- Loads the corresponding CSV file from `/data/`
- Applies Fisher Score + RFE for feature selection
- Trains a CNN model (gender-aware or baseline)
- Outputs evaluation metrics (accuracy, confusion matrix, etc.)

---

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
- Confusion Matrices: Used for visualizing class-wise performance.

## Results Summary

We evaluated the model on three benchmark datasets under two configurations:
- **CNN with Fisher Score only** (baseline)
- **Gender-aware CNN + Fisher Score + RFE** (proposed method)

| Split Type       | Dataset | Configuration         | Model                       | Accuracy (%) |
|------------------|---------|------------------------|-----------------------------|--------------|
| 80–20            | RAVDESS | Both genders           | CNN + Fisher + RFE          | 55.21        |
| 80–20            | RAVDESS | Female only            | Gender + CNN + Fisher + RFE | **68.53**    |
| 80–20            | RAVDESS | Male only              | Gender + CNN + Fisher + RFE | 63.89        |
| 80–20            | EMO-DB  | Both genders           | CNN + Fisher + RFE          | 78.50        |
| 80–20            | EMO-DB  | Female only            | Gender + CNN + Fisher + RFE | **85.25**    |
| 80–20            | EMO-DB  | Male only              | Gender + CNN + Fisher + RFE | 80.43        |
| 80–20            | EMOVO   | Both genders           | CNN + Fisher + RFE          | 75.21        |
| 80–20            | EMOVO   | Female only            | Gender + CNN + Fisher + RFE | **86.21**    |
| 80–20            | EMOVO   | Male only              | Gender + CNN + Fisher + RFE | 84.75        |
| Leave-one-out    | RAVDESS | Both genders           | CNN + Fisher + RFE          | 46.67        |
| Leave-one-out    | RAVDESS | Female only            | Gender + CNN + Fisher + RFE | 63.33        |
| Leave-one-out    | RAVDESS | Male only              | Gender + CNN + Fisher + RFE | 56.67        |
| Leave-one-out    | EMO-DB  | Both genders           | CNN + Fisher + RFE          | 71.43        |
| Leave-one-out    | EMO-DB  | Female only            | Gender + CNN + Fisher + RFE | **77.94**    |
| Leave-one-out    | EMO-DB  | Male only              | Gender + CNN + Fisher + RFE | 74.55        |
| Leave-one-out    | EMOVO   | Both genders           | CNN + Fisher + RFE          | 48.98        |
| Leave-one-out    | EMOVO   | Female only            | Gender + CNN + Fisher + RFE | 37.76        |
| Leave-one-out    | EMOVO   | Male only              | Gender + CNN + Fisher + RFE | 35.71        |

- All evaluations use **5-fold cross-validation**.
- Gender-aware preprocessing improves accuracy consistently across all datasets.
