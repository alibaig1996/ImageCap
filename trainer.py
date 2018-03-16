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
	def __init__(self, wordIndexes, dimFeature=4096, dimEmbed=300, dimHidden=300, nTimeStep=16, nEpochs=10, batchSize=100, learningRate=0.01):

		# Model parameters
		self.wordIndexes = wordIndexes
		self.D = dimFeature
		self.vocabSize = len(wordIndexes)
		self.dimEmbed = dimEmbed
		self.dimHidden = dimHidden
		self.nTimeStep = nTimeStep
		self._start = word_to_idx['<START>']
		self._null = word_to_idx['<NULL>']

		self.weightInitializer = tf.contrib.layers.xavier_initializer()
		self.constInitializer = tf.constant_initializer(0.0)

		self.features = tf.placeholder(tf.float32, [None, self.L, self.D])
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
	def buildCaptionEmbeddings(self, inputs):
		with tf.variable_scope('captionEmbedding'):
			captionEmbedding = tf.get_variable('captionEmbedding', [self.vocabSize, self.dimEmbed], initializer=self.emb_initializer)
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

	def buildLogits(self, hiddenState):
		with tf.variable_scope('logits'):
			hiddenWeights = tf.get_variable('hiddenWeights', [self.H, self.M], initializer = self.weightInitializer)
			hiddenBiases = tf.get_variable('hiddenBiases', [self.M], initializer=self.constInitializer)

			outWeights = tf.get_variable('outWeights', [self.M, self.M], initializer = self.weightInitializer)
			outBiases = tf.get_variable('outBiases', [self.V], initializer=self.constInitializer)

			hiddenLogits = tf.nn.tanh(tf.matmul(tf.nn.dropout(hiddenState, 0.5), hiddenWeights) + hiddenBiases)

			outLogits = tf.nn.tanh(tf.matmul(tf.nn.dropout(hiddenLogits, 0.5), outWeights) + outBiases)

			return outLogits


	def build():

		features = self.features
		captions = self.captions

		batchSize = self.batchSize

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
		
		embeddedFeatures = self.buildImageEmbeddings(features=normalizedFeatures)

		# cellState, hiddenState =  self.buildInitialStates(inputs=embeddedFeatures)

		embeddedCaptions = self.buildCaptionEmbeddings(inputs = inputSequence)

		loss = 0.0

		lstmCell = tf.nn.rnn_cell.BasicLSTMCell(num_units = self.H)

		zero_state = lstm_cell.zero_state(batch_size=tf.shape(features)[0], dtype=tf.float32)

		for t in range(self.nTimeStep):
			with tf.variable_scope('LSTM', reuse=(t!=0)):
				_, (cellState, hiddenState) = lstmCell(inputs=embeddedCaptions[:,t,:], state = [cellState, hiddenState])

			logits = buildLogits(hiddenState)

			loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=outputSequence[:, t]) * outputMask[:, t])

		return (loss / tf.to_float(batchSize))