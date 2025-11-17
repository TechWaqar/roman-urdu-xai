# Explainable AI for Roman Urdu Offensive Language Detection

## 🎯 Project Overview
This project implements an explainable AI system for detecting offensive language in Roman Urdu text using BERT (Multilingual) and explainability techniques (LIME/SHAP). Built as part of BS Information Technology final year project at Sindh Agriculture University.

## 🚀 Live Demo
**Try it here:** [Roman Urdu XAI Web App](https://share.streamlit.io) *(link will be added after deployment)*

## 📊 Features
- Offensive language detection with 86%+ accuracy
- Real-time LIME-based word-level explanations
- Interactive web interface built with Streamlit
- Support for Roman Urdu code-mixed text
- Trained on HS-RU-20 dataset

## 🛠️ Setup
1. Create virtual environment: `python -m venv venv`
2. Activate: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/Mac)
3. Install dependencies: `pip install -r requirements.txt`
4. Run the app: `streamlit run streamlit_app.py`

## 📁 Project Structure
```
├── data/raw/          # Original datasets
├── data/processed/    # Cleaned data
├── notebooks/         # Jupyter experiments
├── src/               # Source code
├── results/           # Models & metrics
├── streamlit_app.py   # Web application
└── requirements.txt   # Dependencies
```

## 📝 Usage
1. Place raw data in `data/raw/`
2. Run preprocessing: `python src/preprocessing/clean_text.py`
3. Train models: Follow notebooks 00 → 01 → 02 → 03
4. Launch web app: `streamlit run streamlit_app.py`

## 🎓 Author
**Waqar Ahmed**  
BS Information Technology Student  
Sindh Agriculture University

📧 [waqarahm@gmail.com](mailto:waqarahm@gmail.com)  
💼 [LinkedIn](https://www.linkedin.com/in/waqar-ahmed-researcher)  
🐙 [GitHub](https://github.com/TechWaqar)

## 🏆 Certifications
- Google IT Support Professional
- Google AI Essentials
- Cisco Networking Essentials
- Microsoft Office 365

## 📄 License
This project is open source and available for educational purposes.
