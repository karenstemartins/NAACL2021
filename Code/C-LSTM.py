# based on: https://github.com/EngSalem/TextClassification_Off_the_shelf
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
import sklearn
import os
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors

from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.callbacks import EarlyStopping
from keras.layers import InputLayer, Conv1D, GlobalMaxPooling1D, LSTM, MaxPooling1D, TimeDistributed
from keras import backend as K
from keras.regularizers import l2

number_of_filter = 100
filter_length = 3
memory_dimension = 200
batch_size = 32

for n_run in range(5):

    name_writer = 'clstm_experts' + '_number_of_filter_' + str(number_of_filter) + '_filter_length_' +str(filter_length) + '_memory_dimension_' + str(memory_dimension) + '_batch_size_' + str(batch_size) + '_i_' + str(n_run)
    train_file_name = 'expert_train.csv'
    val_file_name = 'expert_val.csv'


    if os.path.exists('model/' + name_writer + '.h5'):
        print(fName)
    else:

        train_data = pd.read_csv(train_file_name)
        val_data = pd.read_csv(val_file_name)

        #convert metascore to positive (>60) and negative (<40)
        labels_train = train_data.metascore.apply(lambda x:0 if float(x) >= 50 else 1)
        labels_val = val_data.metascore.apply(lambda x:0 if float(x) >= 50 else 1)

        #select only review and eliminate numbers
        reviews = train_data.review.str.replace('\d+', '')

        NUM_WORDS=90000000000 # the maximum number of words to keep, based on word frequency. Only the most common num_words-1 words will be kept.
        tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'', lower=True)
        tokenizer.fit_on_texts(reviews) #create a array of words with indices
        NUM_WORDS = len(tokenizer.word_index) #new number
        sequences_train = tokenizer.texts_to_sequences(reviews)  #transform in an array of indices
        sequences_val = tokenizer.texts_to_sequences(val_data.review) #transform in an array of indices
        word_index = tokenizer.word_index

        print('Found %s unique tokens.' % len(word_index))

        #add pad in train and val
        X_train = pad_sequences(sequences_train, maxlen=256)
        X_val = pad_sequences(sequences_val, maxlen=X_train.shape[1])

        #load google embeddings
        word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        EMBEDDING_DIM=300
        vocabulary_size=min(len(word_index)+1,NUM_WORDS+1)
        embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))

        #inicialize embedding matrix based on Google embeddings
        for word, i in word_index.items():
            try:
                embedding_vector = word_vectors[word]
                embedding_matrix[i] = embedding_vector
            except KeyError:
                embedding_matrix[i]=np.random.uniform(-0.25, 0.25, EMBEDDING_DIM)

        del(word_vectors)


        sequence_length = X_train.shape[1]
        print("sequence_length", sequence_length)

        print('Build model...')
        model = Sequential()

        model.add(InputLayer(input_shape=(sequence_length,)))
        model.add(Embedding(vocabulary_size,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=sequence_length,
                                    trainable=True))
        model.add(Dropout(0.5))
        model.add(Conv1D(number_of_filter,
                         filter_length,
                         padding='valid',
                         activation='relu',
                         strides=1))
        model.add(LSTM(memory_dimension))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.summary()

        epochs = 5
        val_loss = {'loss': 1., 'epoch': 0}
        val_acc = 0
        for e in range(0,epochs):
            print('epochs', e)

            hist = model.fit(X_train, labels_train, validation_data=(X_val, labels_val), batch_size=batch_size, epochs=1)

            if hist.history['val_loss'][0] < val_loss['loss']:
                val_acc = hist.history['val_acc'][0]
                val_loss = {'loss': hist.history['val_loss'][0], 'epoch': e}
                model.save_weights('model/' + name_writer + '.h5', overwrite=True)
                print('epochs_save', e)
