import re
import string
import pandas as pd
from pathlib import Path

def clean_text(text):
    """
    Basic text cleaning for Roman Urdu
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags (keep the word)
    text = re.sub(r'#', '', text)
    
    # Remove emojis (optional - comment out if you want to keep them)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def normalize_roman_urdu(text):
    """
    Normalize common Roman Urdu spelling variations
    """
    normalizations = {
        'achha': 'acha',
        'aacha': 'acha',
        'bht': 'bohat',
        'bhot': 'bohat',
        'kr': 'kar',
        'krna': 'karna',
        'kro': 'karo',
        'hy': 'hai',
        'hain': 'hain',
        'haan': 'han',
        'nhi': 'nahi',
        'nai': 'nahi',
        'yar': 'yaar',
        'yr': 'yaar',
        'ap': 'aap',
        'apna': 'apna',
        'tmara': 'tumhara',
        'tmhari': 'tumhari',
    }
    
    words = text.split()
    normalized_words = [normalizations.get(word, word) for word in words]
    return ' '.join(normalized_words)

def preprocess_dataset(input_csv, output_csv, text_column='text', label_column='label'):
    """
    Preprocess entire dataset
    """
    print(f"Loading dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"Original shape: {df.shape}")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=[text_column])
    
    # Remove null values
    df = df.dropna(subset=[text_column, label_column])
    
    print(f"After cleaning: {df.shape}")
    
    # Clean text
    print("Cleaning text...")
    df['cleaned_text'] = df[text_column].apply(clean_text)
    
    # Normalize Roman Urdu
    print("Normalizing Roman Urdu...")
    df['cleaned_text'] = df['cleaned_text'].apply(normalize_roman_urdu)
    
    # Remove empty texts
    df = df[df['cleaned_text'].str.len() > 0]
    
    print(f"Final shape: {df.shape}")
    
    # Save processed data
    df.to_csv(output_csv, index=False)
    print(f"Processed data saved to {output_csv}")
    
    return df

if __name__ == "__main__":
    # Example usage
    input_path = "../../data/raw/dataset.csv"
    output_path = "../../data/processed/cleaned_dataset.csv"
    
    preprocess_dataset(input_path, output_path)