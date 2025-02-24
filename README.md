# Deep Learning–Based Prediction of Major Adverse Cardiovascular Events (MACE) After Liver Transplantation

**Repository:** [Tao-AI-group/MCI-to-AD-Progression-Model](https://github.com/Tao-AI-group/LT_MACE_Prediction)

## Overview

This repository provides a deep learning framework to predict **Major Adverse Cardiovascular Events (MACE)** in patients after **Liver Transplantation (LT)**. It is based on methods described in the paper:

> **Deep Learning–Based Prediction Modeling of Major Adverse Cardiovascular Events After Liver Transplantation**  
> *Abdelhameed, A., Bhangu, H., Feng, J., Li, F., Hu, X., Patel, P., Yang, L., & Tao, C. (2024)

This project aims to showcase the end-to-end pipeline:
1. Data preprocessing  
2. Exploratory data analysis  
3. Model training & evaluation  
4. Result interpretation & visualization


## Project Structure

This repository is structured to facilitate access to the various components of the MACE post liver tranplantation prediction model. Below is an overview of the directory and file organization:
- **/Preprocessing/**: Contains scripts for preprocessing inputs to the models.
  - **01-Preprocessing_Input.ipynb**: This script is used for preprocess patient claims, create aggregated encounters and create MACE labels within prediction intervals.
  - **02-Preprocessing_Diagnoses_Medications_Procedures.ipynb**: This script preprocess the patients diagnoses, medications and procedures data.
  - **03-Generate_Model_Input.py**: The script generates final model input.
