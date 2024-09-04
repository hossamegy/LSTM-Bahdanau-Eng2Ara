from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Attention, Concatenate, Dense

def seq2seq_NMT(vocab_size_en, vocab_size_ar, embedding_dim=50):
    # encoder
    encoder_inputs = Input(shape=(None,))
    encoder_embedding = Embedding(
        input_dim=vocab_size_en,
        output_dim=embedding_dim,
        embeddings_initializer="uniform",
        mask_zero= True
    )(encoder_inputs)

    encoder_lstm = LSTM(256, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None,))
    decoder_embedding = Embedding(        
        input_dim=vocab_size_ar,
        output_dim=embedding_dim,
        embeddings_initializer="uniform",
        mask_zero= True
    )(decoder_inputs)
    
    decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

    # Attention Layer
    attention = Attention()
    attention_output = attention([decoder_outputs, encoder_outputs])

    # Concatenating attention output and decoder LSTM output 
    decoder_concat_input = Concatenate(axis=-1)([decoder_outputs, attention_output])

    # Dense layer
    decoder_dense = Dense(vocab_size_ar, activation='softmax')
    decoder_outputs = decoder_dense(decoder_concat_input)

    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model