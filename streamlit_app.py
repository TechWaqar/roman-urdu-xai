# streamlit_app.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from lime.lime_text import LimeTextExplainer

# Page config
st.set_page_config(page_title="Roman Urdu XAI", page_icon="🤖", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);}
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    MODEL_PATH = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
LABELS = ["Offensive", "Neutral"]

# LIME explainer
@st.cache_resource
def get_explainer():
    return LimeTextExplainer(class_names=LABELS)

explainer = get_explainer()

def predict_proba(texts):
    predictions = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            predictions.append(probs[0].cpu().numpy())
    return np.array(predictions)

# UI
st.title("🤖 Roman Urdu Offensive Language Detection")
st.markdown("**AI-Powered Text Analysis with Explainable AI**")

st.markdown("---")

# Input
text_input = st.text_area("Enter Roman Urdu Text:", height=150, 
                          placeholder="Example: yaar ye bilkul ghatiya hai...")

if st.button("🚀 Analyze with AI", use_container_width=True):
    if text_input.strip():
        with st.spinner("Analyzing..."):
            # Prediction
            inputs = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=128, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Display results
            st.markdown("### 📊 Analysis Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Prediction</h4>
                    <h2 style="color: {'#ef4444' if LABELS[predicted_class]=='Offensive' else '#10b981'}">
                        {LABELS[predicted_class]}
                    </h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Confidence</h4>
                    <h2 style="color: #667eea">{confidence*100:.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Category</h4>
                    <h2>{LABELS[predicted_class]}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # LIME Explanation
            st.markdown("### 🔍 Word-Level Explanation")
            with st.spinner("Generating LIME explanation..."):
                exp = explainer.explain_instance(text_input, predict_proba, num_features=10, num_samples=100)
                lime_weights = dict(exp.as_list())
                
                # Display highlighted text
                words = text_input.split()
                html_output = ""
                for word in words:
                    weight = lime_weights.get(word.lower(), 0)
                    if abs(weight) < 0.1:
                        color = "#10b981"
                        bg = "rgba(16, 185, 129, 0.2)"
                    elif weight > 0.1:
                        color = "#ef4444"
                        bg = "rgba(239, 68, 68, 0.2)"
                    else:
                        color = "#f59e0b"
                        bg = "rgba(245, 158, 11, 0.2)"
                    
                    html_output += f'<span style="background:{bg}; color:{color}; padding:4px 8px; margin:2px; border-radius:8px; font-weight:600;">{word}</span> '
                
                st.markdown(html_output, unsafe_allow_html=True)
                
                st.markdown("""
                <div style="margin-top:1rem; text-align:center;">
                    <span style="background:rgba(16,185,129,0.2); color:#10b981; padding:4px 12px; margin:4px; border-radius:8px;">🟢 Safe</span>
                    <span style="background:rgba(245,158,11,0.2); color:#f59e0b; padding:4px 12px; margin:4px; border-radius:8px;">🟡 Mild</span>
                    <span style="background:rgba(239,68,68,0.2); color:#ef4444; padding:4px 12px; margin:4px; border-radius:8px;">🔴 Offensive</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some text to analyze!")

# Sidebar
# Sidebar
with st.sidebar:
    # Profile Image (add your image to project folder as 'profile.jpg')
    try:
        st.image("profile.jpg", width=150)
    except:
        st.markdown("### 👨‍🎓")
    
    st.markdown("### About Author")
    st.info("""
    **Waqar Ahmed**  
    
    BS Information Technology student at Sindh Agriculture University with a passion for AI and Machine Learning.
    
    **Certifications:**
    - Google IT Support Professional
    - Google AI Essentials
    - Microsoft Office 365 Training
    - Cisco Networking Essentials
    
    **Notable Projects:**
    - Roman Urdu Offensive Language Detection (NLP)
    - IT Diagrams with Lucidchart
    - Word Counter App (Java)
    
    Actively exploring cutting-edge technologies in Natural Language Processing and Explainable AI for low-resource languages.
    """)
    
    st.markdown("### 🔗 Connect with Me")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:waqarahm@gmail.com)")
    with col2:
        st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/waqar-ahmed-researcher/)")
    with col3:
        st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/TechWaqar)")
    
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
    - Python 3.10+
    - PyTorch & Transformers
    - BERT (Multilingual)
    - LIME & SHAP
    - Streamlit
    - Java

    """)
