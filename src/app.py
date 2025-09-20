import pandas as pd

def load_data(file_path):
    try:
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from '{file_path}'.")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None

def get_data_head(df, n=10):
    # Use the .head() method to get the first n rows
    return df.head(n)


def check_missing_values(df):
    # Your code here: Return the sum of null values for each column
    return df.isnull().sum()

def check_for_empty_strings(df):
    # Your code here: Find and return the count of empty strings in the 'Text' column
    return (df['Text'].str.strip() == '').sum()


def get_language_distribution(df):
    # Your code here: Return the value counts of the 'Language' column
    return df['Language'].value_counts()

# --- Main execution block ---
if __name__ == "__main__":
    DATA_FILE_PATH = 'Language Detection.csv'
    
    # 1. Load the data
    data = load_data(DATA_FILE_PATH)
    
    if data is not None:
        # 2. Get and display the head
        print("--- First 10 Rows of the Dataset ---")
        head_df = get_data_head(data, n=10)
        print(head_df)
        print("\n" + "="*50 + "\n")

        # 3. Check for missing values
        print("--- Missing Values Check ---")
        missing_values = check_missing_values(data)
        print(missing_values)
        if missing_values.sum() == 0:
            print("--> Verdict: No missing (null) values found.")
        print("\n" + "="*50 + "\n")

        # 4. Check for empty strings in the 'Text' column
        print("--- Empty String Check in 'Text' Column ---")
        empty_count = check_for_empty_strings(data)
        print(f"Number of rows with empty or whitespace-only text: {empty_count}")
        if empty_count == 0:
            print("--> Verdict: No empty text entries found.")
        print("\n" + "="*50 + "\n")

        # 5. Get and display language distribution
        print("--- Language Distribution (Top 10) ---")
        lang_dist = get_language_distribution(data)
        print(lang_dist.head(10)) # Print top 10 for brevity
        print(f"\nTotal unique languages: {len(lang_dist)}")
        print("\n" + "="*50 + "\n")
