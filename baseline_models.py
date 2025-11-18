from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_svm(X_train, y_train, C=1.0):
    """
    Train SVM classifier
    """
    model = LinearSVC(C=C, random_state=42, max_iter=5000)
    model.fit(X_train, y_train)
    return model

def train_naive_bayes(X_train, y_train):
    """
    Train Naive Bayes classifier
    """
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train, C=1.0):
    """
    Train Logistic Regression
    """
    model = LogisticRegression(C=C, random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, n_estimators=100):
    """
    Train Random Forest
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath):
    """
    Save trained model
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """
    Load saved model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model