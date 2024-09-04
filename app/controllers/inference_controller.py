from .preprocessing_controller import PreprocessingController
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

class InferenceController:

    def translate(self, sentence_en: str, en_tokenizer, ar_tokenizer, model) -> str:
        """
        Translates an English sentence to another language using the provided model.
        
        Parameters:
        - sentence_en: The input sentence in English to translate.
        - en_tokenizer: The tokenizer used for the English language.
        - ar_tokenizer: The tokenizer used for the target language.
        - model: The translation model to use for generating predictions.
        
        Returns:
        - The translated sentence.
        """
        MAX_LENGTH = 40  # Maximum length of the input and output sequences
        PADDING_TYPE = 'post'  # Padding type to use ('pre' or 'post')
        translation = ""
        
        # Preprocess the input sentence for the model
        X = PreprocessingController().preprocess_input(sentence_en, en_tokenizer, MAX_LENGTH, PADDING_TYPE)
        
        for word_idx in range(MAX_LENGTH):
            # Prepare the decoder input sequence with 'sos' (start of sentence)
            X_dec = ar_tokenizer.texts_to_sequences(["sos " + translation])
            X_dec = pad_sequences(X_dec, maxlen=MAX_LENGTH, padding=PADDING_TYPE)
            
            # Predict the next word probabilities
            y_proba = model.predict((X, X_dec))[0, word_idx]
            
            # Find the index of the most probable word
            predicted_word_id = np.argmax(y_proba)
            
            # Check if the predicted word is 'eos' (end of sentence)
            if predicted_word_id == ar_tokenizer.word_index['eos']:
                break
            
            # Convert the predicted word index to the actual word
            predicted_word = ar_tokenizer.index_word[predicted_word_id]
            
            # Append the predicted word to the translation
            translation += " " + predicted_word
        
        # Reverse the words in the translation and return the result
        return " ".join(translation.strip().split()[::-1])
