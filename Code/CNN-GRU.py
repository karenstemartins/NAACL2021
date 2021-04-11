# based on: https://github.com/ultimate010/crnn
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
import sklearn
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
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dropout, Activation, Flatten, \
    Embedding, Convolution1D, MaxPooling1D, AveragePooling1D, \
    Input, Dense, concatenate, merge
from keras.regularizers import l2
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.constraints import maxnorm
from keras.datasets import imdb
from keras import callbacks
from keras.utils import generic_utils
from keras.models import Model
from keras.optimizers import Adadelta
import time
import os
import numpy

number_of_filter = 100
filter_length = 5
rnn_output_size = 150
pool_size = 3
batch_size = 16

for n_run in range(5):

    name_writer = 'cnn_gru_experts' + '_number_of_filter_' + str(
        number_of_filter) + '_filter_length_' + str(
        filter_length) + '_memory_dimension_' + str(
        rnn_output_size) + '_pool_size_' + str(pool_size) + '_batch_size_' + str(batch_size) + '_i_' + str(n_run)

    train_file_name = 'expert_train.csv'
    val_file_name = 'expert_val.csv'

    if os.path.exists('model/' + name_writer + '.h5'):
        print(fName)
    else:

        train_data = pd.read_csv(train_file_name)
        val_data = pd.read_csv(val_file_name)

        # convert metascore to positive and negative
        labels_train = train_data.metascore.apply(lambda x: 0 if float(x) >= 5 else 1)
        labels_val = val_data.metascore.apply(lambda x: 0 if float(x) >= 5 else 1)

        reviews = train_data.review.str.replace('\d+', '')

        NUM_WORDS = 90000000000  # the maximum number of words to keep, based on word frequency. Only the most common num_words-1 words will be kept.
        tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'', lower=True)
        tokenizer.fit_on_texts(reviews)  # create a array of words with indices
        NUM_WORDS = len(tokenizer.word_index)  # new number
        sequences_train = tokenizer.texts_to_sequences(reviews)  # transform in an array of indices
        sequences_val = tokenizer.texts_to_sequences(val_data.review)  # transform in an array of indices
        word_index = tokenizer.word_index

        print('Found %s unique tokens.' % len(word_index))

        # add pad in train and val
        X_train = pad_sequences(sequences_train, maxlen=512)
        X_val = pad_sequences(sequences_val, maxlen=X_train.shape[1])

        # load google embeddings
        word_vectors = KeyedVectors.load_word2vec_format(
            'GoogleNews-vectors-negative300.bin', binary=True)
        EMBEDDING_DIM = 300
        vocabulary_size = min(len(word_index) + 1, NUM_WORDS + 1)
        embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))

        # inicialize embedding matrix based on Google embeddings
        for word, i in word_index.items():
            try:
                embedding_vector = word_vectors[word]
                embedding_matrix[i] = embedding_vector
            except KeyError:
                embedding_matrix[i] = np.random.uniform(-0.25, 0.25, EMBEDDING_DIM)

        del (word_vectors)

        sequence_length = X_train.shape[1]
        print("sequence_length", sequence_length)


        hidden_dims = number_of_filter * 2
        RNN = GRU

        print('Build model...')

        main_input = Input(shape=(sequence_length,), dtype='int32', name='main_input')

        embedding_k = Embedding(vocabulary_size,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    trainable=True,
                              name='embedding')(main_input)

        embedding_k = Dropout(0.50)(embedding_k)

        conv4 = Convolution1D(nb_filter=number_of_filter,
                              filter_length=filter_length,
                              border_mode='valid',
                              activation='relu',
                              subsample_length=1,
                              name='conv4')(embedding_k)

        maxConv4 = MaxPooling1D(pool_length=pool_size,
                                name='maxConv4')(conv4)

        conv5 = Convolution1D(nb_filter=number_of_filter,
                              filter_length=filter_length,
                              border_mode='valid',
                              activation='relu',
                              subsample_length=1,
                              name='conv5')(embedding_k)
        maxConv5 = MaxPooling1D(pool_length=pool_size,
                                name='maxConv5')(conv5)

        x = concatenate([maxConv4, maxConv5])

        x = Dropout(0.15)(x)

        x = RNN(rnn_output_size)(x)

        x = Dense(hidden_dims, activation='relu', init='he_normal',
                  W_constraint=maxnorm(3), b_constraint=maxnorm(3),
                  name='mlp')(x)

        x = Dropout(0.10, name='drop')(x)

        output = Dense(1, init='he_normal',
                       activation='sigmoid', name='output')(x)

        model = Model(input=main_input, output=output)
        model.compile(loss={'output': 'binary_crossentropy'},
                      optimizer=Adadelta(lr=0.95, epsilon=1e-06),
                      metrics=["accuracy"])

        model.summary()
        epochs = 5
        val_loss = {'loss': 1., 'epoch': 0}
        val_acc = 0

        for e in range(0,epochs):
            print('epochs', e)


            hist = model.fit(X_train, labels_train, validation_data=(X_val, labels_val),
                             batch_size=batch_size, epochs=1)


            if hist.history['val_loss'][0] < val_loss['loss']:
                val_acc = hist.history['val_acc'][0]
                val_loss = {'loss': hist.history['val_loss'][0], 'epoch': e}
                model.save_weights('model/' + name_writer + '.h5', overwrite=True)
                print('epochs_save', e)