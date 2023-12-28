import streamlit as st
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set threshold for classification
threshold = 0.5

def predict(text):
    # Tokenize text
    inputs = tokenizer(text, return_tensors="pt")

    # Make prediction
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits).item()

    # Apply threshold
    confidence = torch.sigmoid(logits).max().item()
    prediction = 1 if confidence >= threshold else 0

    return prediction, confidence

def main():
    # Center-aligned title using HTML and CSS
    st.markdown("""
        <style>
            .title {
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='title'>Spam Message Detection</h1>", unsafe_allow_html=True)

    # Sidebar for file upload
    uploaded_file = st.file_uploader("Upload a Dataset CSV file", type="csv")

    # Main content
    user_input = st.text_input("Enter a message:")
    if st.button("Predict Message"):
        if user_input:
            prediction, confidence = predict(user_input)
            bot_response = "SPAM" if prediction == 1 else "NOT SPAM"
            st.text(f"Bot Response: {bot_response}")
            st.text(f"Confidence: {confidence}")
        else:
            st.warning("Please enter a message.")

    if uploaded_file is not None:
        try:
            # Read CSV file with explicit encoding specification
            df = pd.read_csv(uploaded_file, encoding='latin1')

            # Allow user to select the text column
            text_column = st.selectbox("Select the text column:", df.columns)

            # Predict button for the selected column
            if st.button("Predict Column"):
                # Make prediction for each text in the selected column
                predictions = []
                confidences = []
                for text in df[text_column].astype(str):  # Ensure the text is a string
                    prediction, confidence = predict(text)
                    predictions.append(prediction)
                    confidences.append(confidence)

                # Display results
                df['Prediction'] = predictions
                df['Confidence'] = confidences
                st.dataframe(df)
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
