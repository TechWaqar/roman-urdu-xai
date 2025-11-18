from src.preprocessing.clean_text import preprocess_dataset

# Process RUHSOLD dataset
df = preprocess_dataset(
    input_csv='data/raw/ruhsold_dataset.csv',
    output_csv='data/processed/ruhsold_cleaned.csv',
    text_column='text',  # ← Change if different
    label_column='label',  # ← Change if different
    extract_features=True
)