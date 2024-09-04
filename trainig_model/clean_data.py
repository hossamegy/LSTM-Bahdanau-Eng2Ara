import pandas as pd
import string
import re

def clean_dataset():
    read_merge_df = pd.read_csv(r"data\merge_df.csv")

    df = pd.DataFrame({
        'encoder_in': read_merge_df.english.tolist(),
        'decoder_in': read_merge_df.arabic.tolist(),
        'decoder_out': read_merge_df.arabic.tolist()
    })

    exclude = set(string.punctuation)
    remove_digits = str.maketrans('', '', string.digits)

    def preprocess_sentences(sentence, input_type='encoder_in'):
        sentence = sentence.lower()
        sentence = sentence.replace("'", '')
        sentence = ''.join(ch for ch in sentence if ch not in exclude)
        sentence = sentence.translate(remove_digits)
        sentence = sentence.strip()
        sentence = re.sub(" +", " ", sentence)
        if input_type == 'encoder_in':
            return sentence
        elif input_type == 'decoder_in':
            sentence = sentence.split()
            sentence = sentence[::-1] 
            sentence = ' '.join(sentence)
            sentence = "sos" + " " + sentence
            return sentence

        elif input_type == 'decoder_out':
            sentence = sentence.split()
            sentence = sentence[::-1] 
            sentence = ' '.join(sentence)
            sentence =  sentence + " "+ "eos"
            return sentence

    df['encoder_in'] = df['encoder_in'].apply(lambda x: preprocess_sentences(x, input_type='encoder_in'))
    df['decoder_in'] = df['decoder_in'].apply(lambda x: preprocess_sentences(x,  input_type='decoder_in'))
    df['decoder_out'] = df['decoder_out'].apply(lambda x: preprocess_sentences(x,  input_type='decoder_out'))

    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(r"data\clean_merge_df.csv", index=False)