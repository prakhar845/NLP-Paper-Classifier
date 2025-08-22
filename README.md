# Multi-Label Text Classification of Scientific Papers

## Project Overview
In the age of information, the rapid growth of scientific literature presents a significant challenge: how can we efficiently organize and navigate this vast sea of knowledge? This project addresses this challenge by developing an intelligent system to automatically classify scientific papers into multiple subject categories using only the text from their abstracts.

This repository documents an end-to-end Natural Language Processing (NLP) solution, starting with a classical machine learning baseline and advancing to a more sophisticated deep learning architecture. The final, high-performing model is deployed as a simple, interactive web application, demonstrating a complete workflow from raw data to a practical, real-world tool.

### Technology Stack
1. **Languages & Libraries:** Python, Pandas, NLTK, Scikit-learn, TensorFlow, Keras, Gradio
2. **Techniques:** Text Preprocessing, TF-IDF Vectorization, Multi-Label Classification, Deep Learning
3. **Models:** Logistic Regression (Baseline), Long Short-Term Memory (LSTM) Network
4. **Deployment:** Gradio for creating an interactive web UI

### Project Structure
NLP_Paper_Classifier/
|
|-- tcs_env/                  # Virtual environment (ignored by Git)
|-- arxiv_data.csv            # Raw data file (ignored by Git)
|-- arxiv_classifier_model.h5 # Trained Keras model (ignored by Git)
|-- tokenizer.pickle          # Saved tokenizer object (ignored by Git)
|-- mlb.pickle                # Saved binarizer object (ignored by Git)
|-- analysis.ipynb            # Jupyter Notebook with the full analysis and model training
|-- app.py                    # Python script for the Gradio web application
|-- .gitignore                # Specifies files to be ignored by Git
|-- README.md                 # This file
|-- requirements.txt          # List of Python dependencies

### Methodology
The project was executed through a structured, five-step process, ensuring a robust and well-evaluated final product.

1. **Data Preprocessing:** The foundation of any NLP model is clean, well-structured data. The model was trained on a dataset of over 50,000 scientific paper abstracts from arXiv. The raw text was meticulously cleaned by converting it to lowercase, removing punctuation and common English stopwords, and tokenizing it for the model. A critical step for this multi-label task was transforming the categorical labels (e.g., cs.AI, cs.LG) into a multi-hot encoded binary format using Scikit-learn's MultiLabelBinarizer, making them suitable for training.

2. **Baseline Model:** To properly evaluate the effectiveness of a deep learning approach, it's essential to first establish a benchmark. A baseline model was created using classical machine learning techniques:
   
  1. **TF-IDF Vectorization:** Text was converted into numerical features representing the importance of each word.
  2. **OneVsRest Classifier:** This strategy allowed a binary classifier (Logistic Regression) to be used for the multi-label problem by training one classifier per label. This baseline provided a solid F1-score to measure against, ensuring that the more complex deep learning model offered a significant performance improvement.
     
3. **Advanced Model Architecture (LSTM)** To capture the nuanced meaning and contextual relationships within the abstracts, an advanced model was built using a Long Short-Term Memory (LSTM) network, a type of Recurrent Neural Network (RNN) perfectly suited for sequential data like text. The Keras model architecture consists of:

   1. **An Embedding Layer:** Converts words into dense numerical vectors, capturing semantic relationships.
   2. **An LSTM Layer:** Processes the sequence of word vectors, learning long-range dependencies and the contextual flow of the text.
   3. **A Dense Output Layer:** Uses a sigmoid activation function to output a probability score for each possible category, allowing a single abstract to be assigned multiple labels.

4. **Performance Evaluation** The models were rigorously evaluated using the micro-averaged F1-score, a metric well-suited for multi-label classification tasks, especially those with potential class imbalance. As hypothesized, the LSTM model's ability to understand context and word order resulted in a significant performance improvement over the TF-IDF baseline, confirming the value of a deep learning approach for this task.

6. **Deployment as an Interactive Web App** A model's true utility is realized when it can be easily used. The final step of this project was to deploy the trained LSTM model into a practical, interactive web application. Using the Gradio library, a simple user interface was created that allows anyone—researchers, students, or librarians—to paste in a scientific abstract and receive instant, real-time category predictions. This demonstrates the full end-to-end lifecycle, from raw data to a tangible, deployed AI tool.

### How to Run
**Clone the repository:**

git clone https://github.com/prakhar845/NLP-Paper-Classifier.git
cd NLP-Paper-Classifier

**Set up the environment:**

python -m venv tcs_env
source tcs_env/bin/activate  # On Windows: tcs_env\Scripts\activate
pip install -r requirements.txt

1. **Download the data:** Download the "ArXiv Paper Abstracts" dataset from Kaggle and place arxiv_data.csv in the root directory.

2. **Train the model:** Open and run the analysis.ipynb Jupyter Notebook to train the model and save the necessary files (.h5, .pickle).

### Launch the app:
python app.py

**Open the local URL provided in your browser to use the application.**

## Contributing
Contributions are welcome! If you have suggestions for improvements, new analysis ideas, or bug fixes, please:

1. Fork the repository.
2. Create a new branch (git checkout -b feature/your-feature-name).
3. Make your changes and commit them (git commit -m 'Add new feature').
4. Push to the branch (git push origin feature/your-feature-name).
5. Open a Pull Request.
