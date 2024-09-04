import pandas as pd
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def preprocessing(MAX_LENGTH, PADDING_TYPE, OOV_TOK):
    read_clean_df = pd.read_csv(r"data\clean_merge_df.csv")

    en_sentences = read_clean_df['encoder_in'].to_list()
    ar_sentences =  read_clean_df['decoder_in'].to_list() + read_clean_df['decoder_out'].to_list()

    en_tokenizer = Tokenizer(oov_token=OOV_TOK)
    en_tokenizer.fit_on_texts(en_sentences)
    en_vocab = en_tokenizer.word_index

    ar_tokenizer = Tokenizer(oov_token=OOV_TOK)
    ar_tokenizer.fit_on_texts(ar_sentences)
    ar_vocab = ar_tokenizer.word_index

    encoder_sequences = en_tokenizer.texts_to_sequences(read_clean_df['encoder_in'])
    decoder_in_sequences = ar_tokenizer.texts_to_sequences(read_clean_df['decoder_in'])
    decoder_out_sequences = ar_tokenizer.texts_to_sequences(read_clean_df['decoder_out'])

    final_encoder_in = pad_sequences(
        encoder_sequences,
        maxlen=MAX_LENGTH,
        padding=PADDING_TYPE,
    )

    final_decoder_in = pad_sequences(
        decoder_in_sequences,
        maxlen=MAX_LENGTH,
        padding=PADDING_TYPE,
    )

    final_decoder_out = pad_sequences(
        decoder_out_sequences,
        maxlen=MAX_LENGTH,
        padding=PADDING_TYPE,
    )
    dic = {
        "size_en_vocab":int(len(en_vocab) + 1),
        "size_ar_vocab":int(len(ar_vocab) + 1),
        "final_encoder_in":final_encoder_in,
        "final_decoder_in":final_decoder_in,
        "final_decoder_out":final_decoder_out
    }
    
    # Save the tokenizer to a JSON file
    tokenizer_json = en_tokenizer.to_json()
    with open('en_tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    en_tokenizer_json = en_tokenizer.to_json()
    with open('en_tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(en_tokenizer_json, ensure_ascii=False))
    
    ar_tokenizer_json = ar_tokenizer.to_json()
    with open('ar_tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(ar_tokenizer_json, ensure_ascii=False))
    return dic

preprocessing(    MAX_LENGTH=40, 
    TRUNC_TYPE="post",
    PADDING_TYPE="post",
    OOV_TOK="<OOV>")