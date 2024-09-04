import tensorflow as tf
from keras.preprocessing.text import tokenizer_from_json
import json

# Disable GPU usage for TensorFlow
tf.config.set_visible_devices([], 'GPU')

class LoadModel:
    def __init__(self):
        """
        Initializes the LoadModel class by loading tokenizers and the model.
        """
        # Load the English tokenizer from a JSON file
        with open(r'model\en_tokenizer.json', 'r', encoding='utf-8') as f:
            en_tokenizer_json = json.load(f)  # Read the JSON data from the file
            self.en_tokenizer = tokenizer_from_json(en_tokenizer_json)  # Convert JSON data to a tokenizer object

        # Load the Arabic tokenizer from a JSON file
        with open(r'model\ar_tokenizer.json', 'r', encoding='utf-8') as f:
            ar_tokenizer_json = json.load(f)  # Read the JSON data from the file
            self.ar_tokenizer = tokenizer_from_json(ar_tokenizer_json)  # Convert JSON data to a tokenizer object

        # Load the model from an H5 file
        self.model = tf.keras.models.load_model(r'model\model_ntm-v32.h5')  # Load the pre-trained model
