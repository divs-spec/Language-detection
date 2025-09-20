import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import os

def train_and_save_model(data_path, model_path):
    # Step 1: Load the data from data_path
    try:
        data = pd.read_csv(data_path)
        data.dropna(inplace=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return None

    # Step 2: Define features (X) and labels (y)
    X = data['Text']
    y = data['Language']

    # Step 3: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Step 4: Create a Scikit-learn Pipeline
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1, 2))), # Using n-grams can improve performance
        ('classifier', MultinomialNB())
    ])

    # Step 5: Train the model
    print("Training the model...")
    pipeline.fit(X_train, y_train)

    # Step 6: Evaluate the model and get accuracy
    print("Evaluating the model...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Step 7: Save the trained pipeline to model_path
    print(f"Saving the model to {model_path}...")
    joblib.dump(pipeline, model_path)
    
    # Step 8: Return the accuracy
    return accuracy    

# --- Main execution block ---
if __name__ == "__main__":
    # Define file paths
    CLEAN_DATA_FILE = 'Language_Detection_Cleaned.csv'
    MODEL_OUTPUT_FILE = 'language_detection_model.joblib'

    model_accuracy = train_and_save_model(CLEAN_DATA_FILE, MODEL_OUTPUT_FILE)

    if model_accuracy is not None:
        print("\n--- Training Complete ---")
        print(f"Model saved to: {MODEL_OUTPUT_FILE}")
        print(f"Model Accuracy on Test Data: {model_accuracy:.4f}")
    else:
        print("\nModel training failed. Please check the error messages above.")


    print("\n--- Testing the saved model with sample inputs ---")
    if os.path.exists(MODEL_OUTPUT_FILE):
        loaded_model = joblib.load(MODEL_OUTPUT_FILE)

        sample_inputs = [
            "This is a test of the language detection system.", # English
            "Ceci est un test du système de détection de langue.", # French
            "Este es un sistema de prueba de detección de idioma.", # Spanish
            "Это тест системы определения языка.", # Russian
            "هذا اختبار لنظام كشف اللغة", # Arabic
            "Questo è un test del sistema di rilevamento della lingua.", # Italian
            "Dit is een test van het taaldetectiesysteem." # Dutch
        ]

        # Make predictions
        predictions = loaded_model.predict(sample_inputs)

        # Print the results
        for i, text in enumerate(sample_inputs):
            print(f"Input: '{text}'")
            print(f"--> Predicted Language: {predictions[i]}\n")
    else:
        print(f"Error: Model file '{MODEL_OUTPUT_FILE}' not found. Cannot run demonstration.")
