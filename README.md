# 📱 SMS Spam Detection System using NLP

![SMS Spam Detection](https://img.shields.io/badge/SMS-Spam%20Detection-blue?logo=python&logoColor=white)

A machine learning-based SMS spam detection system using Natural Language Processing (NLP). This project leverages a Naive Bayes classifier and TF-IDF vectorization to classify SMS messages as either spam or ham (non-spam). The system is powered by Streamlit, providing an interactive user interface to test the model with new SMS inputs.

---

## 🛠️ Features

- **Interactive UI:** Easily classify SMS messages as spam or ham using Streamlit.
- **Preprocessing:** Removes stop words and performs text normalization (lowercasing, removing non-alphabetical characters).
- **Model Training:** Naive Bayes classifier trained on the SMS Spam Collection dataset.
- **Performance Evaluation:** Accuracy score for model evaluation.
- **Customizable:** Users can input their own SMS messages for classification.

## 🚀 How to Use

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/sms-spam-detection-nlp.git

2. Navigate to the project directory:
   ```bash
   cd sms-spam-detection-nlp

3. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Mac/Linux
   venv\Scripts\activate     # For Windows

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

5. Run the Streamlit app:
   ```bash
   streamlit run sms_spam_detection.py

6. Open the app in your browser and input SMS messages for classification!

## 📚 Dataset

- **The project uses the SMSSpamCollection dataset from the UCI Machine Learning Repository. The dataset consists of SMS messages classified as either spam or ham**.

## 🤖 Model

- **Model**: Naive Bayes (MultinomialNB)
- **Feature Extraction**: TF-IDF Vectorization
- **Accuracy**: The model is evaluated using accuracy score after testing on the dataset.

## 🖥️ Requirements

- **Python 3.x**
- **pandas**
- **numpy**
- **sklearn**
- **nltk**
- **streamlit**
- **requests**
  
- Install dependencies by running:
   ```bash
   pip install -r requirements.txt

## 📸 Screenshots

![test 1](https://github.com/user-attachments/assets/45969022-9b75-4512-ae5d-bcb2c6193306)

![test 2](https://github.com/user-attachments/assets/78ddb150-1ddb-4b9b-b5fc-a69b7817381d)

## ⚙️ Technologies Used

- **Python**: The primary language used for data processing and model training.
- **Streamlit**: To build the interactive web interface for SMS classification.
- **Scikit-learn**: For machine learning model training and evaluation.
- **NLTK**: For text preprocessing (stopwords, text cleaning).
- **TF-IDF Vectorizer**: For converting text data into numerical vectors.
- **Naive Bayes Classifier**: To classify messages as spam or ham.

## 💡 Contributing

- Feel free to fork this project, submit issues, and send pull requests. Any contributions are welcome!

## 🤝 Contact

- For any questions or suggestions, feel free to contact me:

- 📧 [Email](mailto:gauravghandat12@gmail.com)
- 💼 [LinkedIn](www.linkedin.com/in/gaurav-ghandat-68a5a22b4)
