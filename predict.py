import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_new_text(text, tokenizer, max_length):
  """Preprocesses new text for sentiment analysis prediction."""
  sequences = tokenizer.texts_to_sequences([text])
  padded = pad_sequences(sequences, maxlen=max_length, padding='post')
  return padded

def predict(model_path="model.h5", text="", tokenizer_path="tokenizer.joblib"):
  """Predicts sentiment on a new text review.
  """
  try:
    # Load the pre-trained model by benski
    model = tf.keras.models.load_model(model_path)
    if tokenizer_path:
      with open(tokenizer_path, 'rb') as handle:
        tokenizer = joblib.load(handle)
    else:
      raise ValueError("Tokenizer required for prediction.")

    # Preprocess the user input
    max_length = model.layers[0].output_shape[1]
    user_input_padded = preprocess_new_text(text, tokenizer, max_length)

    # prediction
    prediction = model.predict(user_input_padded)
    sentiment = 'Positive' if prediction[0] >= 0.5 else 'Niggative'
    print(f"Sentiment: {sentiment}| Confidence: {prediction[0]} ")

  except FileNotFoundError as e:
    print(f"Error: Model file not found ({e}).")

if __name__ == "__main__":
  model_path = "model.h5"
  tokenizer_path = "tokenizer.joblib"

  # Prompt user for input
  text = input("Enter a review to analyze: ")

  # Make prediction
  predict(model_path, text, tokenizer_path)
