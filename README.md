# Modality Ablation Study for Alzheimer Disease Detection

This repository explores an ablation study for detecting Alzheimer's Disease (AD) using various combinations of input modalities. The project focuses on evaluating the impact of different modality combinations (Electronic Health Records (EHR), Single Nucleotide Polymorphisms (SNP), Magnetic Resonance Imaging (MRI), and Nicara features) on the performance of a hybrid model designed for AD detection. The study aims to identify the most robust modality combinations for Alzheimer's detection.

## Overview

The goal of this project is to conduct a modality ablation study that evaluates how different combinations of modalities (EHR, SNP, MRI, Nicara features) affect the performance of a deep learning model in predicting Alzheimer's disease. We adapt a hybrid model capable of handling any combination of these modalities, allowing us to experiment with all possible modality combinations.

The final objective is to create a report on the best modality combinations that produce the most reliable and accurate results in AD detection.

## Project Structure

This repository contains:

- **Data Processing**: Scripts to preprocess and merge the modalities into a format suitable for input into the hybrid model.
- **Hybrid Model**: A flexible neural network model that can accept any combination of the four modalities (EHR, SNP, MRI, Nicara features).
- **Ablation Study**: Code to systematically test all combinations of the modalities and report on the performance of each.
- **Results**: A detailed report summarizing the results of the ablation study, highlighting the best-performing combinations.

## Modalities Used

1. **EHR (Electronic Health Records)**: Structured medical history data, such as diagnoses, medications, lab results, and clinical notes.
2. **SNP (Single Nucleotide Polymorphisms)**: Genetic data capturing variations in the DNA sequence, which may be associated with the risk of Alzheimer's disease.
3. **MRI (Magnetic Resonance Imaging)**: Imaging data providing insights into the brain's structure and potential signs of neurodegeneration.
4. **Nicara Features**: Derived features from Nicara, representing a mix of data derived from multiple sources, potentially including behavioral and environmental factors.

## Installation

To use this repository, clone it to your local machine:

```bash
git clone https://github.com/planetlibrary/Modality-Combo-Benchmark.git
cd Modality-Combo-Benchmark
pip install -r requirements.txt
python train_model.py
```
