# src/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from lime.lime_text import LimeTextExplainer

app = Flask(__name__)
CORS(app)

# Load your trained model
MODEL_PATH = "results/models/bert_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Class labels (swapped to fix the issue)
LABELS = ["Offensive", "Neutral"]

# Initialize LIME explainer
explainer = LimeTextExplainer(class_names=LABELS)

def predict_proba(texts):
    """Prediction function for LIME"""
    predictions = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            predictions.append(probs[0].cpu().numpy())
    return np.array(predictions)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Get prediction
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Generate LIME explanation
        print("Generating LIME explanation...")
        exp = explainer.explain_instance(
            text, 
            predict_proba, 
            num_features=10,
            num_samples=100
        )
        
        # Get word importance from LIME
        lime_weights = dict(exp.as_list())
        
        # Split text into words and assign colors based on LIME scores
        words = text.split()
        word_explanations = []
        
        for word in words:
            word_lower = word.lower()
            # Find the weight for this word (LIME might have slightly different tokenization)
            weight = lime_weights.get(word_lower, 0)
            
            # Classify importance based on weight
            if abs(weight) < 0.1:
                importance = 'safe'  # Low importance
            elif weight > 0.1:
                importance = 'offensive'  # Positive weight = supports offensive
            else:
                importance = 'mild'  # Negative weight = supports neutral
            
            word_explanations.append({
                'word': word,
                'importance': importance,
                'weight': float(weight)
            })
        
        print(f"LIME explanation generated: {len(word_explanations)} words")
        
        return jsonify({
            'prediction': LABELS[predicted_class],
            'confidence': round(confidence * 100, 2),
            'category': LABELS[predicted_class],
            'word_explanations': word_explanations
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'Model loaded successfully'})

if __name__ == '__main__':
    print("Starting Flask server with LIME explanations...")
    print("Model loaded from:", MODEL_PATH)
    app.run(debug=True, port=5000)