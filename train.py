import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from model import define_model

def load_dataset():
    #dataset from keras kase we broke
    imdb, _ = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
    return imdb['train'], imdb['test']

def preprocess_data(data, vocab_size, max_length, trunc_type='post', oov_tok='<OOV>'):
    
    sentences = [str(s.numpy()) for s, _ in data]
    
    labels = [l.numpy() for _, l in data]

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    
    
    padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

    return padded, np.array(labels), tokenizer


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel('Epochs')
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

def train_model(vocab_size=10000, embedding_dim=16, max_length=120, trunc_type='post', oov_tok='<OOV>', num_epochs=10):
    
    
    
    """
    hyperparameters:
    vocab_size=10000 this means that the tokenizer will only consider the 10000 most common words,
    embedding_dim=16 this is the dimension of the embedding vectors which means that each word will be represented by a vector of 16 dimensions; reason why 16 is because it is a common value yun lang HAHAHHAA
    max_length=120 this is the maximum length of the sequences; this is the length of the padded sequences which in laymans term is the length of the sentences
    trunc_type='post' this is the truncation type which means that if the sentence is longer than the max_length, the extra words will be truncated from the end of the sentence
    oov_tok='<OOV>' this is the out-of-vocabulary token which is used to replace words that are not in the tokenizer's word index
    num_epochs=10 this is the number of epochs for training the model pero di naman to kumpleto because we have early stopping callback to stop overfitting
    """
    
    
    train_data, test_data = load_dataset()
    
    
    # test_data.head()
    
    
    
    train_padded, train_labels, tokenizer = preprocess_data(train_data, vocab_size, max_length, trunc_type, oov_tok)
    test_padded, test_labels, _ = preprocess_data(test_data, vocab_size, max_length, trunc_type, oov_tok)
    
    
    
    model = define_model(vocab_size, embedding_dim, max_length)




    # Oversample the negative class (after preprocessing)
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    train_padded, train_labels = ros.fit_resample(train_padded, train_labels)


#early stopping cause why not diba 
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4)

    history = model.fit(train_padded, train_labels, epochs=num_epochs,
                        validation_data=(test_padded, test_labels),callbacks=early_stopping)
    
    
    
    acc, val_acc = model.evaluate(test_padded, test_labels, verbose=0)
    print('> Test Accuracy: %.3f' % (_ * 100))
    print('> accuracy: %.3f' % (acc * 100))
    print('> val_accuracy: %.3f' % (val_acc * 100))
    plot_graphs(history, 'loss')
    plot_graphs(history, 'accuracy')
    
    model.save("model.h5")
    print("Model saved to model.h5")
    return tokenizer

if __name__ == "__main__":
    tokenizer = train_model()
    from collections import Counter

    train_data, test_data = load_dataset()

    train_labels = [l.numpy() for _, l in train_data]

    # count ng positive and negative labels
    label_counts = Counter(train_labels)
    print(f"Class Distribution: {label_counts}")
    # Save the tokenizer 
    tokenizer_path = "tokenizer.joblib"
    with open(tokenizer_path, 'wb') as handle:
        joblib.dump(tokenizer, handle)
        
        
