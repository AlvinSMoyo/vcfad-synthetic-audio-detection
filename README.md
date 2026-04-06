# VCFAD: Synthetic Audio Detection Pipeline

An end-to-end machine learning pipeline for generating synthetic speech and detecting fake audio using classical machine learning and deep learning approaches.

## Project Overview

This project explores the detection of synthetic (fake) audio by building a modular pipeline across three stages:

1. **Data Preparation**  
   Preparing and validating real speech datasets for downstream modelling.

2. **Synthetic Audio Generation**  
   Generating synthetic speech samples using a pretrained Text-to-Speech (TTS) model.

3. **Fake Audio Detection**  
   Training and comparing multiple models to classify real vs synthetic speech.

This project was developed as an **independent technical dry run inspired by a VCFAD-style brief**, and should be interpreted as a proof-of-concept rather than a production-grade or mentor-reviewed implementation.

---

## Repository Structure

```bash
VCFAD_Notebook_1_Data_Preparation.ipynb
VCFAD_Notebook_2_Synthetic_Audio_Generation.ipynb
VCFAD_Notebook_3_Synthetic_Audio_Detection.ipynb
README.md
requirements.txt
```

---

## Notebook Breakdown

### Notebook 1 — Data Preparation
This notebook establishes the project’s data layer by:

- loading and validating real speech data
- structuring metadata for downstream tasks
- preparing clean subsets for both generation and detection

### Notebook 2 — Synthetic Audio Generation
This notebook generates synthetic speech samples using **SpeechT5**.

> **Important note:**  
> This implementation uses a fixed speaker embedding and does **not** perform true multi-speaker voice cloning. It is more accurately described as **synthetic speech generation using TTS**, which is sufficient for building the downstream fake-audio detector.

### Notebook 3 — Synthetic Audio Detection
This notebook trains and compares multiple models for real-vs-fake classification, including:

- **XGBoost**
- **Random Forest**
- **CNN on Mel-Spectrograms**

---

## Tech Stack

- Python
- Jupyter / Google Colab
- LibriSpeech
- SpeechT5
- Whisper (for transcription-based evaluation)
- Librosa
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Matplotlib / Seaborn

---

## Key Learning Themes

- audio data preparation and validation
- synthetic speech generation
- acoustic feature engineering
- fake audio classification
- model comparison across classical ML and CNN architectures
- practical handling of dataset and environment constraints

---

## Project Context

This project is positioned as an **independent pre-project technical build** designed to explore the feasibility of a synthetic speech detection system.

It is intended to demonstrate:

- applied machine learning workflow design
- modular pipeline thinking
- fraud / impersonation detection relevance
- practical experimentation in audio AI

---

## Author

**Alvin Siphosenkosi Moyo**  
Applied AI | Data Science | ML Systems | Finance Domain Expertise

---

## Status

**Completed as an MVP / technical dry run**  
Further enhancements could include:

- multi-speaker voice cloning
- stronger speaker similarity evaluation
- robustness testing across multiple TTS generators
- deployment as an interactive demo or API
