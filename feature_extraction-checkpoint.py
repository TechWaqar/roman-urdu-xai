from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle

def create_tfidf_features(texts, max_features=10000, ngram_range=(1, 2)):
    """
    Create TF-IDF features
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.8
    )
    
    features = vectorizer.fit_transform(texts)
    return features, vectorizer

def save_vectorizer(vectorizer, filepath):
    """
    Save vectorizer for later use
    """
    with open(filepath, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved to {filepath}")

def load_vectorizer(filepath):
    """
    Load saved vectorizer
    """
    with open(filepath, 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer