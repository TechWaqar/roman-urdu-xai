# Explainable AI for Roman Urdu Offensive Language Detection

## Project Overview
This project implements an explainable AI system for detecting offensive language in Roman Urdu text using deep learning and XAI techniques (LIME/SHAP).

## Setup
1. Create virtual environment: `python -m venv venv`
2. Activate: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/Mac)
3. Install dependencies: `pip install -r requirements.txt`
4. Download NLTK data: Run `python src/utils/download_nltk.py`

## Project Structure
- `data/raw/` - Original datasets
- `data/processed/` - Cleaned and preprocessed data
- `notebooks/` - Jupyter notebooks for experiments
- `src/` - Source code modules
- `results/` - Model outputs and metrics
- `docs/` - Documentation and paper drafts

## Usage
1. Place raw data in `data/raw/`
2. Run preprocessing: `python src/preprocessing/clean_text.py`
3. Open Jupyter: `jupyter lab`
4. Follow notebooks in order: 00_setup → 01_baseline → 02_deep_learning → 03_explainability

## Author
Waqar Ahmed - BSIT Student