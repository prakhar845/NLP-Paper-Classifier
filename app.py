import gradio as gr
import tensorflow as tf
from keras.utils import pad_sequences
import pickle
import re
from nltk.corpus import stopwords

model = tf.keras.models.load_model('arxiv_classifier_model.h5', compile=False)

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('mlb.pickle', 'rb') as handle:
    mlb = pickle.load(handle)

print("Model and preprocessing objects loaded successfully.")

stop_words = set(stopwords.words('english'))
max_length = 200

def clean_text(text):
    """Applies the same cleaning from your notebook."""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def predict_categories(abstract):
    """The core function that takes raw text and returns predictions for Gradio."""
    if not abstract.strip():
        return {}

    cleaned_abstract = clean_text(abstract)
    
    seq = tokenizer.texts_to_sequences([cleaned_abstract])
    padded_seq = pad_sequences(seq, maxlen=max_length, padding='post')
    
    pred_probs = model.predict(padded_seq)[0]
    
    top_indices = pred_probs.argsort()[-5:][::-1]
    
    confidences = {mlb.classes_[i]: float(pred_probs[i]) for i in top_indices}
    
    return confidences


iface = gr.Interface(
    fn=predict_categories,
    inputs=gr.Textbox(lines=15, label="Abstract", placeholder="Paste a scientific paper abstract here..."),
    outputs=gr.Label(num_top_classes=5, label="Predicted Categories"),
    title="🔬 Scientific Paper Classifier",
    description="Enter the abstract of a scientific paper to predict its subject categories. The model will return the top 5 most likely categories with their confidence scores.",
    examples=[
        ["We study the problem of multi-label text classification in the context of scientific paper categorization. Our approach leverages a recurrent neural network with long short-term memory (LSTM) cells to capture long-range dependencies in the text. We evaluate our model on the arXiv dataset, demonstrating significant improvements over traditional methods like TF-IDF with logistic regression."]
    ]
)

iface.launch()