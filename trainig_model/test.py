import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
import numpy as np
from keras.preprocessing.text import tokenizer_from_json
import json
tf.config.set_visible_devices([], 'GPU')

# Load the tokenizer from the JSON file
with open(r'model\en_tokenizer.json', 'r', encoding='utf-8') as f:
    en_tokenizer_json = json.load(f)
    en_tokenizer = tokenizer_from_json(en_tokenizer_json)

with open(r'model\ar_tokenizer.json', 'r', encoding='utf-8') as f:
    ar_tokenizer_json = json.load(f)
    ar_tokenizer = tokenizer_from_json(ar_tokenizer_json)

# Assuming these are the sizes and components used in your training model

# Load the model
model = tf.keras.models.load_model(r"model\model_ntm-v32.h5")

def preprocess_en(sentence):
    sentence = sentence.lower()
    sentence = sentence.replace('.', '')
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    return sentence

def preprocess_input(sentence, en_tokenizer, MAX_LENGTH, PADDING_TYPE):
    sentence = preprocess_en(sentence)
    sequence = en_tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, padding=PADDING_TYPE)
    return padded_sequence

def translate(sentence_en):
    MAX_LENGTH = 40
    PADDING_TYPE = 'post' 
    translation = ""
    # Preprocess input sentence
    X = preprocess_input(sentence_en, en_tokenizer, MAX_LENGTH, PADDING_TYPE)
    
    for word_idx in range(MAX_LENGTH):
        X_dec = ar_tokenizer.texts_to_sequences(["sos " + translation])
        X_dec = pad_sequences(X_dec, maxlen=MAX_LENGTH, padding=PADDING_TYPE)
        
        y_proba = model.predict((X, X_dec))[0, word_idx]
        predicted_word_id = np.argmax(y_proba)
        
        if predicted_word_id == ar_tokenizer.word_index['eos']:
            break
        
        predicted_word = ar_tokenizer.index_word[predicted_word_id]
        translation += " " + predicted_word
        
    return " ".join(translation.strip().split()[::-1]) 

# test sample data
sample_english = [
    "i love you",
    "can you help me",
    "i will go to school",
    "do not worry",
    "tell me a story"
] 

for i in sample_english:
    print(f"English sentence: {i}\nArabic sentence:{translate(i)}")
