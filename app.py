import streamlit as st
from transformers import pipeline
import torch

# Load pre-trained summarization model from Hugging Face
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

def summarize_text(text, user_type='casual'):
    """
    Summarize the text based on user preferences.
    - user_type can be 'casual' or 'expert'.
    """
    
    # Customize summary based on user preference
    if user_type == 'expert':
        # For expert: Summarize with a focus on technical aspects
        summary = summarizer(text, max_length=250, min_length=150, do_sample=False)
        summary_text = summary[0]['summary_text']
        # Add more technical content or explanation (this can be refined based on requirements)
        technical_details = "This section covers more in-depth analysis of the technical details."
        return summary_text + "\n" + technical_details
    
    elif user_type == 'casual':
        # For casual reader: Summarize with a focus on key takeaways
        summary = summarizer(text, max_length=150, min_length=100, do_sample=False)
        summary_text = summary[0]['summary_text']
        # Add easy-to-understand explanation (this can be refined based on requirements)
        key_takeaways = "Key takeaways: Focus on the main ideas and high-level overview."
        return summary_text + "\n" + key_takeaways
    
    else:
        return "Invalid user type. Choose 'casual' or 'expert'."


# Streamlit UI
st.title("Personalized Text Summarizer")

# Input text area for the user
text_content = st.text_area("Enter the long-form content (article, paper, etc.)", height=300)

# Dropdown to select user type (Casual or Expert)
user_type = st.selectbox("Select your preference", ["casual", "expert"])

# Summarization button
if st.button("Generate Summary"):
    if text_content:
        # Call the summarize function with the user-selected type
        summary = summarize_text(text_content, user_type)
        st.subheader("Personalized Summary")
        st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")
