from prepare_data import prepare_dataset
from clean_data import clean_dataset
from preprocessing import preprocessing
from buildModel import seq2seq_NMT
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

prepare_dataset()
clean_dataset()

dic = preprocessing(
    MAX_LENGTH=40, 
    PADDING_TYPE="post",
    OOV_TOK="<OOV>"
)
vocab_size_en=dic["size_en_vocab"]
vocab_size_ar=dic['size_ar_vocab']

print(vocab_size_en)
print(vocab_size_ar)

final_encoder_in = dic["final_encoder_in"]
final_decoder_in = dic["final_decoder_in"]
final_decoder_out = dic["final_decoder_in"]

model = seq2seq_NMT(
    vocab_size_en=vocab_size_en,
    vocab_size_ar=vocab_size_ar,
    embedding_dim=100
 )

checkpoint_path = "model_ntm-v3.h5"

checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    save_freq='epoch',
    period=1
)

model.fit(
    [final_encoder_in, final_decoder_in], 
     final_decoder_out, 
     batch_size=16, 
     epochs=1500,
     callbacks=[checkpoint_callback],  
)
