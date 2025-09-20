import pandas as pd
import re 

def preprocess_text(text):
    # 1. Convert text to lowercase
    text = text.lower()
    
    # 2. Remove bracketed content (like [1], [2], etc.) and other symbols
    # This regex removes anything inside square brackets.
    text = re.sub(r'\[.*?\]', '', text)
    
    # 3. Remove punctuation, numbers, and any character that is not a letter or whitespace
    # [^\w\s] matches any character that is not a word character (a-z, A-Z, 0-9, _) or whitespace.
    # We add \d to also remove numbers explicitly if they are part of words.
    text = re.sub(r'[^\w\s]|\d', '', text)
    
    # 4. Remove extra whitespace (including newlines and tabs) and strip leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# --- Main execution block ---
if __name__ == "__main__":
    # Define file paths
    INPUT_FILE = 'Language Detection.csv'
    OUTPUT_FILE = 'Language_Detection_Cleaned.csv'
    
    print("Starting data cleaning process...")
    
    # 1. Load the data
    try:
        data = pd.read_csv(INPUT_FILE)
        print(f"Successfully loaded {INPUT_FILE}. Shape: {data.shape}")
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Please place it in the correct directory.")
        exit()

    # 2. Handle missing values
    # We drop rows with any null values to ensure clean processing.
    data.dropna(inplace=True)
    print(f"Data shape after dropping missing values: {data.shape}")
    
    # 3. Apply the cleaning function to the 'Text' column
    # We ensure the 'Text' column is treated as a string to prevent errors with .apply()
    print("Cleaning text data...")
    data['Text'] = data['Text'].astype(str).apply(preprocess_text)
    
    # Optional: Remove any rows that might have become empty after cleaning
    data = data[data['Text'] != '']
    print(f"Data shape after removing empty text rows: {data.shape}")

    # 4. Save the cleaned data to a new CSV file
    # We only need the 'Text' and 'Language' columns for our model.
    try:
        data.to_csv(OUTPUT_FILE, columns=['Text', 'Language'], index=False)
        print(f"Data cleaning complete. Cleaned data saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")
