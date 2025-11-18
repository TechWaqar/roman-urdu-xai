"""
Roman Urdu Text Preprocessing Module - Enhanced Version
Combines basic cleaning with advanced normalization and feature extraction
"""

import re
import json
import pandas as pd
from pathlib import Path


# ============================================================================
# BASIC CLEANING FUNCTIONS (Your Original Code - Enhanced)
# ============================================================================

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
    
    # Remove emojis
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
    Enhanced with more variations
    """
    normalizations = {
        # Your original normalizations
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
        
        # Additional common variations
        'kya': 'kya',
        'kia': 'kya',
        'kiya': 'kya',
        'yeh': 'yeh',
        'ye': 'yeh',
        'woh': 'woh',
        'wo': 'woh',
        'mein': 'mein',
        'main': 'mein',
        'me': 'mein',
        'hoon': 'hoon',
        'hun': 'hoon',
        'hon': 'hoon',
        'theek': 'theek',
        'thik': 'theek',
        'teek': 'theek',
        'bhai': 'bhai',
        'bhae': 'bhai',
        'tha': 'tha',
        'thaa': 'tha',
        'kuch': 'kuch',
        'kch': 'kuch',
        'sab': 'sab',
        'sb': 'sab',
        'agar': 'agar',
        'agr': 'agar',
        'lekin': 'lekin',
        'lkin': 'lekin',
        'phir': 'phir',
        'fir': 'phir',
        'abhi': 'abhi',
        'abi': 'abhi',
        'kaise': 'kaise',
        'kese': 'kaise',
        'kse': 'kaise',
        'kyun': 'kyun',
        'kun': 'kyun',
        'kyu': 'kyun',
        'matlab': 'matlab',
        'mtlb': 'matlab',
        'bilkul': 'bilkul',
        'blkl': 'bilkul',
        'shayad': 'shayad',
        'shyd': 'shayad',
        'sirf': 'sirf',
        'srf': 'sirf',
        'jab': 'jab',
        'jb': 'jab',
        'tab': 'tab',
        'tb': 'tab',
    }
    
    words = text.split()
    normalized_words = [normalizations.get(word, word) for word in words]
    return ' '.join(normalized_words)


# ============================================================================
# ADVANCED FEATURES (New Enhanced Functions)
# ============================================================================

def normalize_repeated_chars(text):
    """
    Reduce repeated characters: haaaaa -> haa, loooool -> lool
    Keeps maximum 2 repetitions
    """
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return text


def extract_text_features(text):
    """
    Extract useful features from text before cleaning
    (useful for ML models and analysis)
    
    Returns:
        Dictionary of features
    """
    if not isinstance(text, str):
        return {
            'has_url': False,
            'has_mention': False,
            'has_hashtag': False,
            'has_emoji': False,
            'has_repeated_chars': False,
            'num_exclamation': 0,
            'num_question': 0,
            'num_caps': 0,
            'text_length': 0,
            'word_count': 0,
            'has_numbers': False,
        }
    
    features = {
        'has_url': bool(re.search(r'http\S+|www\S+|https\S+', text)),
        'has_mention': bool(re.search(r'@\w+', text)),
        'has_hashtag': bool(re.search(r'#\w+', text)),
        'has_emoji': bool(re.search(
            r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]', 
            text
        )),
        'has_repeated_chars': bool(re.search(r'(.)\1{2,}', text)),
        'num_exclamation': text.count('!'),
        'num_question': text.count('?'),
        'num_caps': sum(1 for c in text if c.isupper()),
        'text_length': len(text),
        'word_count': len(text.split()),
        'has_numbers': bool(re.search(r'\d', text)),
    }
    return features


# ============================================================================
# DATASET PREPROCESSING (Your Original - Enhanced)
# ============================================================================

def preprocess_dataset(input_csv, output_csv, 
                      text_column='text', 
                      label_column='label',
                      extract_features=False,
                      normalize_repeats=True):
    """
    Preprocess entire dataset with enhanced options
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to save processed CSV
        text_column: Name of text column
        label_column: Name of label column
        extract_features: Whether to extract additional features
        normalize_repeats: Whether to normalize repeated characters
    
    Returns:
        Processed DataFrame
    """
    print("=" * 70)
    print("ROMAN URDU TEXT PREPROCESSING")
    print("=" * 70)
    
    print(f"\n📂 Loading dataset from {input_csv}...")
    try:
        df = pd.read_csv(input_csv, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(input_csv, encoding='latin-1')
        except:
            df = pd.read_csv(input_csv, encoding='cp1252')
    
    print(f"✅ Original shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Check if columns exist
    if text_column not in df.columns:
        print(f"\n⚠️  ERROR: Column '{text_column}' not found!")
        print(f"   Available columns: {df.columns.tolist()}")
        return None
    
    if label_column not in df.columns:
        print(f"\n⚠️  ERROR: Column '{label_column}' not found!")
        print(f"   Available columns: {df.columns.tolist()}")
        return None
    
    # Remove duplicates
    original_size = len(df)
    df = df.drop_duplicates(subset=[text_column])
    duplicates_removed = original_size - len(df)
    print(f"\n🔍 Removed {duplicates_removed} duplicates")
    
    # Remove null values
    df = df.dropna(subset=[text_column, label_column])
    print(f"✅ After removing nulls: {df.shape}")
    
    # Extract features (optional)
    if extract_features:
        print("\n📊 Extracting features...")
        features_list = df[text_column].apply(extract_text_features).tolist()
        features_df = pd.DataFrame(features_list)
        df = pd.concat([df, features_df], axis=1)
        print(f"   Added {len(features_df.columns)} feature columns")
    
    # Clean text
    print("\n🧹 Cleaning text...")
    df['cleaned_text'] = df[text_column].apply(clean_text)
    
    # Normalize repeated characters
    if normalize_repeats:
        print("🔄 Normalizing repeated characters...")
        df['cleaned_text'] = df['cleaned_text'].apply(normalize_repeated_chars)
    
    # Normalize Roman Urdu spellings
    print("📝 Normalizing Roman Urdu spellings...")
    df['cleaned_text'] = df['cleaned_text'].apply(normalize_roman_urdu)
    
    # Remove empty texts
    before_empty = len(df)
    df = df[df['cleaned_text'].str.len() > 0]
    empty_removed = before_empty - len(df)
    print(f"🗑️  Removed {empty_removed} empty texts after cleaning")
    
    print(f"\n✅ Final shape: {df.shape}")
    
    # Show statistics
    print("\n📈 STATISTICS:")
    print(f"   Total samples: {len(df)}")
    print(f"   Avg text length (original): {df[text_column].str.len().mean():.1f} chars")
    print(f"   Avg text length (cleaned): {df['cleaned_text'].str.len().mean():.1f} chars")
    print(f"   Avg word count: {df['cleaned_text'].str.split().str.len().mean():.1f} words")
    
    if label_column in df.columns:
        print(f"\n🏷️  Label distribution:")
        label_counts = df[label_column].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {label}: {count} ({percentage:.1f}%)")
    
    # Save processed data
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\n💾 Processed data saved to: {output_csv}")
    
    # Show sample
    print("\n📄 Sample cleaned texts:")
    for i, (original, cleaned) in enumerate(zip(
        df[text_column].head(3), 
        df['cleaned_text'].head(3)
    ), 1):
        print(f"\n   {i}. Original: {original[:80]}...")
        print(f"      Cleaned:  {cleaned[:80]}...")
    
    print("\n" + "=" * 70)
    print("✅ PREPROCESSING COMPLETE!")
    print("=" * 70)
    
    return df


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_clean(text):
    """
    Quick one-line cleaning function
    
    Usage:
        cleaned = quick_clean("yeh banda bohot acha hai!!!")
    """
    text = clean_text(text)
    text = normalize_repeated_chars(text)
    text = normalize_roman_urdu(text)
    return text


# ============================================================================
# MAIN - Testing & Examples
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ROMAN URDU TEXT CLEANER - TEST MODE")
    print("=" * 70)
    
    # Test individual functions
    test_texts = [
        "yeh banda bohot acha hai!!! 😊",
        "@username tumhara kaam bohot achha hai #great",
        "kya haal hai??? yaaaar check out http://example.com",
        "mein abhi ghar jaa raha hoon",
        "aaaacha yaar tu bht acha hy 😀😀",
    ]
    
    print("\n📝 Testing individual cleaning functions:\n")
    for i, text in enumerate(test_texts, 1):
        print(f"{i}. Original: {text}")
        cleaned = quick_clean(text)
        print(f"   Cleaned:  {cleaned}")
        features = extract_text_features(text)
        print(f"   Features: {features}\n")
    
    # Test dataset preprocessing
    print("\n" + "=" * 70)
    print("To preprocess a dataset, use:")
    print("=" * 70)
    print("""
from src.preprocessing.clean_text import preprocess_dataset

# Basic usage
df = preprocess_dataset(
    input_csv='data/raw/your_dataset.csv',
    output_csv='data/processed/cleaned_dataset.csv',
    text_column='text',
    label_column='label'
)

# With feature extraction
df = preprocess_dataset(
    input_csv='data/raw/your_dataset.csv',
    output_csv='data/processed/cleaned_dataset.csv',
    text_column='text',
    label_column='label',
    extract_features=True
)
    """)
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE!")
    print("=" * 70)