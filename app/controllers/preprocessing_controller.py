from tensorflow.keras.preprocessing.sequence import pad_sequences
import string

class PreprocessingController:
    
    def preprocess_en(self, sentence: str) -> str:
        """
        Preprocesses an English sentence by lowercasing it, 
        removing periods, and stripping punctuation.
        """
        # Convert the sentence to lowercase
        sentence = sentence.lower()
        
        # Remove periods from the sentence
        sentence = sentence.replace('.', '')
        
        # Remove all punctuation characters
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        
        return sentence

    def preprocess_input(self, sentence: str, en_tokenizer, MAX_LENGTH: int, PADDING_TYPE: str) -> list:
        """
        Preprocesses the input sentence for model prediction:
        - Cleans the sentence
        - Converts it to a sequence of integers
        - Pads the sequence to a specified length
        
        Parameters:
        - sentence: The input sentence to preprocess.
        - en_tokenizer: The tokenizer used to convert text to sequences.
        - MAX_LENGTH: The maximum length of the padded sequence.
        - PADDING_TYPE: The type of padding to use ('pre' or 'post').

        Returns:
        - Padded sequence suitable for model input.
        """
        # Clean the sentence
        sentence = self.preprocess_en(sentence)
        
        # Convert the cleaned sentence to a sequence of integers
        sequence = en_tokenizer.texts_to_sequences([sentence])
        
        # Pad the sequence to the specified length
        padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, padding=PADDING_TYPE)
        
        return padded_sequence
