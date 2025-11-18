import lime
from lime.lime_text import LimeTextExplainer
import shap
import matplotlib.pyplot as plt
import numpy as np

def explain_with_lime(model, vectorizer, text, class_names, num_features=10):
    """
    Generate LIME explanation for a prediction
    """
    # Create LIME explainer
    explainer = LimeTextExplainer(class_names=class_names)
    
    # Define prediction function
    def predict_proba(texts):
        vectors = vectorizer.transform(texts)
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(vectors)
        else:
            # For SVM, convert decision function to probabilities
            decisions = model.decision_function(vectors)
            probas = np.exp(decisions) / np.sum(np.exp(decisions), axis=1, keepdims=True)
            return probas
    
    # Generate explanation
    exp = explainer.explain_instance(
        text, 
        predict_proba, 
        num_features=num_features
    )
    
    return exp

def visualize_lime_explanation(exp, save_path=None):
    """
    Visualize LIME explanation
    """
    fig = exp.as_pyplot_figure()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    return fig

def explain_with_shap(model, vectorizer, texts, background_size=100):
    """
    Generate SHAP explanation
    """
    # Create background dataset
    background = vectorizer.transform(texts[:background_size])
    
    # Create SHAP explainer
    explainer = shap.Explainer(model.predict, background)
    
    return explainer

def plot_shap_values(shap_values, features, save_path=None):
    """
    Plot SHAP values
    """
    shap.summary_plot(shap_values, features, show=False)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()