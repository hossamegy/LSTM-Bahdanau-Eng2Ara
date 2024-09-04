# LSTM-Bahdanau-Eng2Ara
Arabic Translation with LSTM and Bahdanau Attention
This repository contains a neural machine translation (NMT) model for translating text from English to Arabic. The model leverages Long Short-Term Memory (LSTM) networks combined with Bahdanau attention mechanism to achieve high-quality translations.

## Project Overview
The goal of this project is to build a robust and efficient translation model using TensorFlow and Keras. The model architecture includes:

1) LSTM (Long Short-Term Memory) Network: Utilized for both the encoder and decoder to handle sequential data and capture contextual dependencies.
2) Bahdanau Attention Mechanism: Improves translation quality by allowing the model to focus on different parts of the input sequence during decoding.
## Features
1) End-to-End Translation: Translates English sentences into Arabic.
2) Bahdanau Attention: Enhances translation performance by dynamically focusing on different parts of the input sentence.
3) LSTM-Based Architecture: Utilizes LSTM units to manage long-range dependencies and improve context understanding.

# Requirements for using the app
1) Dowonload and install anacond from link -> 
https://www.anaconda.com/download/success

2) create new environment using the following command:
```bash
conda create -n Eng2Ara python=3.9    
```  
3) activate the environment using the following command:
```bash
conda activate Eng2Ara
```
4) pip install requirements file using the following command:
```bash
pip install -r requirements.txt
```
5) Run FastApi server

```bash
uvicorn main:app --reload
```
## Usage
#### Once the server is running, interact with the application via the Swagger documentation or test it using the test_page.html file.
#
# Traing model Code

## **1. Data Preparation**

**The prepare_data.py:** 

    script handles initial data preparation. It reads and processes raw text files to create a consolidated dataset in CSV format. This dataset includes pairs of English and Arabic sentences extracted from provided files.

## **2. Data Cleaning**
**The clean_data.py** 

    script focuses on cleaning the dataset to ensure it is suitable for training. This involves lowercasing, punctuation removal, and formatting sentences. The cleaned data is then saved in a new CSV file, which includes additional preprocessing steps to prepare the data for model training.

## **3. Data Preprocessing**
**In preprocessing.py**

    the cleaned dataset undergoes further preprocessing. This step involves tokenization and padding of sequences to convert text data into a format suitable for training the model. Tokenizers for both English and Arabic are created and saved, facilitating the conversion of text into sequences of integers.

## **4. Model Definition**
**The buildModel.py**

    script defines the Seq2Seq model with Bahdanau attention. The model architecture includes:

    Encoder: Processes input sequences and generates context vectors.
    Decoder: Uses context vectors to generate translated sequences.
    Attention Mechanism: Enhances the decoder’s ability to focus on relevant parts of the input sequence.
    The model is compiled and summarized, providing an overview of its structure and parameters.

## **5. test**
**The test.py**

    script demonstrates how to use the trained model for translation. It includes:
    Preprocessing Functions: For preparing input sentences and decoding the model’s output.
    Translation Function: Takes an English sentence as input and generates the corresponding Arabic translation using the trained model.
    This script also includes example translations to showcase the model’s functionality.


# **Conclusion**

The LSTM-Bahdanau-Eng2Ara project represents a significant advancement in neural machine translation by integrating LSTM networks with Bahdanau attention mechanisms. By leveraging these sophisticated techniques, the model provides high-quality translations from English to Arabic, effectively handling complex sequential data and improving context understanding.

With a streamlined setup process, including detailed instructions for environment setup, dependency installation, and running the FastAPI server, users can easily deploy and interact with the translation service. The comprehensive training pipeline—from data preparation and cleaning to model definition and testing—ensures that the model is both robust and effective.

We encourage users to explore the functionality of the translation model, test its capabilities with various English sentences, and contribute to its development. For further enhancements and updates, please refer to the documentation and stay tuned for future improvements.

Thank you for your interest in the LSTM-Bahdanau-Eng2Ara project. Your feedback and contributions are valuable in advancing the field of machine translation.