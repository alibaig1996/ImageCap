from scipy import ndimage
from collections import Counter

import tensorflow as tf
import numpy as np
import pandas as pd
import hickle
import os
import json

from gensim.models import KeyedVectors

class ImageCapModel(object):
	def __init__(self, wordIndexes, dimFeature=4096, dimEmbed=300, dimHidden=300, nTimeStep=16, dropout = 0.7):

		# Model parameters
		self.wordIndexes = wordIndexes
		self.idx_to_word = {i: w for w, i in wordIndexes.items()}
		self.D = dimFeature
		self.vocabSize = len(wordIndexes)
		self.dimEmbed = dimEmbed
		self.dimHidden = dimHidden
		self.nTimeStep = nTimeStep
		self._start = wordIndexes['<START>']
		self._null = wordIndexes['<NULL>']
		self.dropout = dropout

		self.weightInitializer = tf.contrib.layers.xavier_initializer()
		self.constInitializer = tf.constant_initializer(0.0)
		self.embedInitializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)


		self.features = tf.placeholder(tf.float32, [None, self.D])
		self.captions = tf.placeholder(tf.int32, [None, self.nTimeStep + 1])


	# Maps Feature Map into embedding space
	def buildImageEmbeddings(self, features):
		with tf.variable_scope('imageEmbedding') as embeddingScope:
			imageEmbeddings = tf.contrib.layers.fully_connected(
									inputs=features,
									num_outputs=self.dimHidden,
									activation_fn=None,
									weights_initializer=self.weightInitializer,
									biases_initializer=self.constInitializer,
									scope=embeddingScope)

			return imageEmbeddings

	# Maps Captions into embedding space
	def buildCaptionEmbeddings(self, inputs, reuse=False):
		with tf.variable_scope('captionEmbedding', reuse=reuse):
			captionEmbedding = tf.get_variable('captionEmbedding', [self.vocabSize, self.dimEmbed], initializer=self.embedInitializer)
			vec = tf.nn.embedding_lookup(captionEmbedding, inputs, name='word_vector')  # (N, T, M) or (N, M)
			return vec

	def buildInitialStates(self, inputs):
		with tf.variable_scope("initialStates"):
			hiddenWeights = tf.get_variable('hiddenWeights', [self.D, self.H], initializer=self.weightInitializer)
			hiddenBiases = tf.get_variable('hiddenBiases', [self.H], initializer=self.constInitializer)
			hiddenState = tf.nn.tanh(tf.matmul(inputs, hiddenWeights) + hiddenBiases)

			cellWeights = tf.get_variable('cellWeights', [self.D, self.H], initializer=self.weightInitializer)
			cellBiases = tf.get_variable('cellBiases', [self.H], initializer=self.constInitializer)
			cellState = tf.nn.tanh(tf.matmul(inputs, cellWeights) + cellBiases)

			return cellState, hiddenState

	def buildLogits(self, hiddenState, reuse = False):
		with tf.variable_scope('logits', reuse=reuse):
			outWeights = tf.get_variable('outWeights', [self.dimHidden, self.vocabSize], initializer = self.weightInitializer)
			outBiases = tf.get_variable('outBiases', [self.vocabSize], initializer=self.constInitializer)

			outLogits = (tf.matmul(tf.nn.dropout(hiddenState, self.dropout), outWeights) + outBiases)

			return outLogits


	def build(self):

		features = self.features
		captions = self.captions

		batchSize = tf.shape(features)[0]

		inputSequence = captions[:, :self.nTimeStep]      
		outputSequence = captions[:, 1:]  
		outputMask = tf.to_float(tf.not_equal(outputSequence, 0))

		normalizedFeatures = tf.contrib.layers.batch_norm(inputs=features, 
													decay=0.95,
													center=True,
													scale=True,
													is_training=True,
													updates_collections=None,
													scope="normalizedFeatures")
		
		embeddedFeatures = self.buildImageEmbeddings(features = normalizedFeatures)

		# cellState, hiddenState =  self.buildInitialStates(inputs = embeddedFeatures)

		embeddedCaptions = self.buildCaptionEmbeddings(inputs = inputSequence)

		loss = 0.0

		lstmCell = tf.nn.rnn_cell.BasicLSTMCell(num_units = self.dimHidden)

		zero_state = lstmCell.zero_state(batch_size=batchSize, dtype=tf.float32)

		_, (cellState, hiddenState) = lstmCell(embeddedFeatures, zero_state)

		for t in range(self.nTimeStep):
			with tf.variable_scope('LSTM', reuse=(t!=0)):
				_, (cellState, hiddenState) = lstmCell(inputs=embeddedCaptions[:,t,:], state = [cellState, hiddenState])
				
			logits = self.buildLogits(hiddenState, reuse=(t!=0))

			loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=outputSequence[:, t]) * outputMask[:, t])

		return loss / tf.to_float(batchSize)

	def build_sampler(self, max_len=20):
		features = self.features

		# batch normalize feature vectors
		features = tf.contrib.layers.batch_norm(inputs=features, 
													decay=0.95,
													center=True,
													scale=True,
													is_training=False,
													updates_collections=None,
													scope="normalizedFeatures")

		# c, h = self._get_initial_lstm(features=features)
		# features_proj = self._project_features(features=features)


		image_embeddings = self.buildImageEmbeddings(features = features)

		sampled_word_list = []
		alpha_list = []
		beta_list = []
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.dimHidden)

		zero_state = lstm_cell.zero_state(batch_size=tf.shape(features)[0], dtype=tf.float32)

		_, (c, h) = lstm_cell(image_embeddings, zero_state)

		for t in range(max_len):
			if t == 0:
				x = self.buildCaptionEmbeddings(inputs=tf.fill([tf.shape(features)[0]], self._start))
			else:
				x = self.buildCaptionEmbeddings(inputs=sampled_word, reuse=True)  

			# context, alpha = self._attention_layer(features, features_proj, h, reuse=(t!=0))
			# alpha_list.append(alpha)

			# if self.selector:
			#     context, beta = self._selector(context, h, reuse=(t!=0)) 
			#     beta_list.append(beta)

			with tf.variable_scope('lstm', reuse=(t!=0)):
				_, (c, h) = lstm_cell(inputs=x, state=[c, h])

			logits = self.buildLogits(h, reuse=(t!=0))
			sampled_word = tf.argmax(logits, 1)       
			sampled_word_list.append(sampled_word)     

		# alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))     # (N, T, L)
		# betas = tf.transpose(tf.squeeze(beta_list), (1, 0))    # (N, T)
		sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))     # (N, max_len)
		return alpha_list, beta_list, sampled_captions
		# return alphas, betas, sampled_captions