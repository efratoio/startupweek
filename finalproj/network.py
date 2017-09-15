
# coding: utf-8

import numpy as np
# import pandas as pd
import _pickle as cPickle
from collections import defaultdict
import re


from numpy.linalg import norm

import sys
import os

os.environ['KERAS_BACKEND']='tensorflow'

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

from keras.layers.merge import concatenate,multiply
from keras.layers import merge
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, Lambda
from keras.layers import RepeatVector, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.layers import TimeDistributed,Permute,Reshape,Activation
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
import functools
import nltk
nltk.download('punkt')
GLOVE_DIR="../data/glove"


def create_embedding_index():
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Total %s word vectors.' % len(embeddings_index))
    return embeddings_index


def create_embedding_matrix(config,word_index):
    embeddings_index= create_embedding_index();
    embedding_matrix = np.random.random((len(word_index) + 1, config["EMBEDDING_DIM"]))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def create_attention(config,base_layer):
	attention = Dense(1, activation='tanh')(base_layer)
	attention = Flatten()(attention)
	attention = Activation('softmax')(attention)
	attention = RepeatVector(config["RNN_DIM"]*2)(attention)

	attention = Permute([2, 1])(attention)

	representation = multiply([base_layer, attention])
	representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(config["RNN_DIM"]*2,))(representation)
	return representation

def create_network(config,word_index):
	embedding_matrix = create_embedding_matrix(config,word_index)
	embedding_layer = Embedding(len(word_index) + 1,
		config["EMBEDDING_DIM"],
		weights=[embedding_matrix],
		input_length=config["MAX_SENT_LENGTH"],
		trainable=True)
	sentence_input = Input(shape=(config["MAX_SENT_LENGTH"],), dtype='int32')
	embedded_sequences = embedding_layer(sentence_input)

	l_lstm = Bidirectional(LSTM(config["RNN_DIM"],return_sequences=config["VANILA_ATTENTION"]))(embedded_sequences)
	if config["VANILA_ATTENTION"]:

		l_lstm = create_attention(config,l_lstm)
	
	sentEncoder = Model(sentence_input, l_lstm)
	auxiliary_input=None
	turn_input = Input(shape=(config["MAX_SENTS"],config["MAX_SENT_LENGTH"]), dtype='int32')
	turn_encoder = TimeDistributed(sentEncoder)(turn_input)
	l_lstm_sent = Bidirectional(LSTM(config["RNN_DIM"],return_sequences=config["VANILA_ATTENTION"]))(turn_encoder)
	if config["VANILA_ATTENTION"]:
		l_lstm_sent = create_attention(config,l_lstm_sent)
		last_layer=None

	if config["VANILA"]:
		last_layer = l_lstm_sent
		preds = Dense(2, activation='softmax')(last_layer)
		model = Model(turn_input, preds)

	else:
		turnsEncoder = Model(turn_input, l_lstm_sent)
		turnsEncoder = Model(turn_input, l_lstm_sent)


		chats_input = Input(shape=(config["MAX_TURNS"],config["MAX_SENTS"],config["MAX_SENT_LENGTH"]), dtype='int32')

		chats_encoder_layer = TimeDistributed(turnsEncoder)
		chats_encoder = chats_encoder_layer(chats_input)

		if config["PROPS"]:
			auxiliary_input = Input(shape=(config["MAX_TURNS"],config["VEC_SIZE"],), name='aux_input')
			last_layer = Dense(config["RNN_DIM"]*2, activation='relu')(concatenate([chats_encoder, auxiliary_input]))
			last_layer = Bidirectional(LSTM(config["RNN_DIM"],return_sequences=config["TURN_ATTENTION"]))(last_layer)
		else:
			last_layer = Bidirectional(LSTM(config["RNN_DIM"],return_sequences=config["TURN_ATTENTION"]))(chats_encoder)


		if config["TURN_ATTENTION"]:
			last_layer = create_attention(config,last_layer)

		preds = Dense(2, activation='softmax')(last_layer)
		if config["PROPS"]:
			model = Model([chats_input,auxiliary_input], preds)
		else:
			model = Model(chats_input, preds)

	model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])
	print(model.summary())
	return model


def create_rnn(config):

	review_input = Input(shape=(config["MAX_TURNS"],config["VEC_SIZE"]), dtype='float32')
	l_lstm = Bidirectional(LSTM(config["RNN_DIM"]))(review_input)


	preds = Dense(2, activation='softmax')(l_lstm)
	model = Model(review_input, preds)

	model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])
	return model
