# Multi-Label Text Classification of Scientific Papers

## Project Overview
This project develops a deep learning system to automatically assign multiple subject categories (e.g., cs.AI, cs.LG) to scientific papers using only the text from their abstracts. The project demonstrates skills in Natural Language Processing (NLP), from classical machine learning baselines to more advanced deep learning architectures. The final model is deployed in a simple, interactive web application.

### Technical Skills
1. **Languages & Libraries:** Python, Pandas, NLTK, Scikit-learn, TensorFlow, Keras, Gradio
2. **Techniques:** Text Preprocessing, TF-IDF Vectorization, Multi-Label Classification, Deep Learning
3. **Models:** Logistic Regression (Baseline), Long Short-Term Memory (LSTM) Network
4. **Deployment:** Gradio for creating an interactive web UI

### Project Structure
Multi-Label_Text_Classification/
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

### Methodology
1. **Data Preprocessing:** The raw text from over 50,000 paper abstracts was cleaned by converting to lowercase, removing punctuation, and filtering out stopwords. The category labels were transformed into a multi-hot encoded binary format using Scikit-learn's MultiLabelBinarizer.

2. **Baseline Model:** A baseline was established using a TF-IDF vectorizer to convert text into numerical features, followed by a OneVsRestClassifier wrapping a LogisticRegression model. This provided a benchmark F1-score to measure against.

3. **Deep Learning Model:** An advanced model was built using a Keras Sequential architecture, including: 
   1. An Embedding layer to learn dense vector representations of words.
   2. An LSTM layer to process the sequence of words and capture contextual information.
   3. A final Dense output layer with a sigmoid activation function to handle the multi-label predictions.

4. **Evaluation:** The models were evaluated using the micro-averaged F1-score, a suitable metric for multi-label classification tasks. The deep learning model showed a significant performance improvement over the baseline.

5. **Deployment:** The trained Keras model and its associated preprocessing objects (Tokenizer, MultiLabelBinarizer) were saved. A simple web application was built using the Gradio library, allowing users to paste an abstract and receive real-time category predictions.

### How to Run
**Clone the repository:**

git clone https://github.com/prakhar845/NLP-Paper-Classifier.git
cd NLP-Paper-Classifier

**Set up the environment:**

python -m venv tcs_env
source tcs_env/bin/activate  # On Windows: tcs_env\Scripts\activate
pip install -r requirements.txt

**(Note: Create a requirements.txt file by running pip freeze > requirements.txt in your activated environment.)**

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