import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

def define_model(vocab_size=10000, embedding_dim=16, max_length=120):
    model = tf.keras.Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        Bidirectional(LSTM(32)),
        Dense(6, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

    
    
    """
        activations functions:
        
        
        relu: x if x > 0
              0 if x <= 0  
              
        sigmoid: tanh
    
    """