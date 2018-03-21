# from core.solver import CaptioningSolver
# from core.model import CaptionGenerator
from core.utils import *
from core.bleu import *
from model import ImageCapModel

import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import os 
import pickle
from scipy import ndimage

class Trainer(object):
	def __init__(self, model, trainData, valData, batchSize = 100, nEpochs = 20, learningRate = 0.001, 
					logPath = 'log/', modelPath = 'model/lstm', printScore = True, printCaptions = 50, saveEpochs = 4):

		self.model = model
		self.trainData = trainData
		self.valData = valData

		self.batchSize = batchSize
		self.nEpochs = nEpochs
		self.learningRate = learningRate
		self.logPath = logPath
		self.modelPath = modelPath
		self.printScore = printScore
		self.printCaptions = printCaptions
		self.saveEpochs = saveEpochs

		self.optimizer = tf.train.AdamOptimizer

		if not os.path.exists(self.modelPath):
			os.makedirs(self.modelPath)
		if not os.path.exists(self.logPath):
			os.makedirs(self.logPath)

	def train(self):

		nSamples = self.trainData['captions'].shape[0]
		# nSamples = self.trainData['features'].shape[0]

		nItersPerEpoch = int(np.ceil(float(nSamples)/self.batchSize))

		features = self.trainData['features']

		print(features.shape)
		print("")

		captions = self.trainData['captions']

		print(captions.shape)
		print("")

		imageIdxs = self.trainData['image_idxs']

		print(imageIdxs.shape)
		print((imageIdxs))
		print(type(imageIdxs[0]))
		print("")

		valFeatures = self.valData['features']
		nItersVal = int(np.ceil(float(valFeatures.shape[0])/self.batchSize))

		with tf.variable_scope(tf.get_variable_scope()):
			loss = self.model.build()
			# quit()
			# loss = self.model.build()
			tf.get_variable_scope().reuse_variables()
			_, _, generated_captions = self.model.build_sampler(max_len=20)

		with tf.variable_scope(tf.get_variable_scope(), reuse=False):
			optimizer = self.optimizer(learning_rate=self.learningRate)
			gradients = tf.gradients(loss, tf.trainable_variables())
			gradsAndVars = list(zip(gradients, tf.trainable_variables()))
			trainOp = optimizer.apply_gradients(grads_and_vars=gradsAndVars)

		tf.summary.scalar('batch_loss', loss)
		for var in tf.trainable_variables():
			tf.summary.histogram(var.op.name, var)
		for grad, var in gradsAndVars:
			tf.summary.histogram(var.op.name+'/gradient', grad)

		summaryOp = tf.summary.merge_all() 

		print("The number of epoch: %d" %self.nEpochs)
		print("Data size: %d" %nSamples)
		print("Batch size: %d" %self.batchSize)
		print("Iterations per epoch: %d" %nItersPerEpoch)

		config = tf.ConfigProto(allow_soft_placement = True)

		with tf.Session(config=config) as sess:
			tf.global_variables_initializer().run()
			summary_writer = tf.summary.FileWriter(self.logPath, graph=tf.get_default_graph())
			saver = tf.train.Saver(max_to_keep=40)

			prevLoss = -1
			currLoss = 0
			start_t = time.time()

			for e in range(self.nEpochs):
				shuffledIndexes = np.random.permutation(nSamples)

				captions = captions[shuffledIndexes]
				imageIdxs = imageIdxs[shuffledIndexes]

				for i in range(nItersPerEpoch):
					captionsBatch = captions[i*self.batchSize:(i+1)*self.batchSize]

					imageIdxsBatch = imageIdxs[i*self.batchSize:(i+1)*self.batchSize]

					featuresBatch = features[imageIdxsBatch]

					feed_dict = {self.model.features: featuresBatch, self.model.captions: captionsBatch}
					_ ,l = sess.run([trainOp, loss], feed_dict)
					currLoss += l

					if i % 100 == 0:
						summary = sess.run(summaryOp, feed_dict)
						summary_writer.add_summary(summary, e*nItersPerEpoch + i)

					if (i+1) % self.printCaptions == 0:
						print("\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" %(e+1, i+1, l))
						print("Elapsed time: ", time.time() - start_t)
						groundTruths = captions[imageIdxs == imageIdxsBatch[0]]
						decoded = decode_captions(groundTruths, self.model.idx_to_word)
						for j, gt in enumerate(decoded):
							print("Ground truth %d: %s" %(j+1, gt)                    )
						genCaps = sess.run(generated_captions, feed_dict)
						decoded = decode_captions(genCaps, self.model.idx_to_word)
						print("Generated caption: %s\n" %decoded[0])

				print("Previous epoch loss: ", prevLoss)
				print("Current epoch loss: ", currLoss)
				print("Elapsed time: ", time.time() - start_t)
				prevLoss = currLoss
				currLoss = 0

				if self.printScore:
					allGeneratedCaptions = np.ndarray((valFeatures.shape[0], 20))
					for i in range(nItersVal):
						featuresBatch = valFeatures[i*self.batchSize:(i+1)*self.batchSize]
						feed_dict = {self.model.features: featuresBatch}
						generatedCaptions = sess.run(generated_captions, feed_dict=feed_dict)  
						allGeneratedCaptions[i*self.batchSize:(i+1)*self.batchSize] = generatedCaptions

					allDecoded = decode_captions(allGeneratedCaptions, self.model.idx_to_word)
					save_pickle(allDecoded, "./data/val/val.candidate.captions.pkl")
					# scores = evaluate(data_path='./data', split='val', get_scores=True)
					# write_bleu(scores=scores, path=self.model_path, epoch=e)

				# save model's parameters
				if (e+1) % self.saveEpochs == 0:
					saver.save(sess, os.path.join(self.modelPath, 'model'), global_step=e+1)
					print("model-%s saved." %(e+1))


def main():
	trainData = load_coco_data(data_path='./data', split='train')
	valData = load_coco_data(data_path='./data', split='val')

	wordIndex = trainData['word_to_idx']

	model = ImageCapModel(wordIndex, dimFeature=4096, dimEmbed=300, dimHidden=300, nTimeStep=26, dropout = 0.7)

	trainer = Trainer(model, trainData, valData, batchSize = 50, nEpochs = 20, learningRate = 0.001, 
					logPath = 'log2/', modelPath = 'model/lstm2', printScore = True, printCaptions = 50, saveEpochs = 4)

	trainer.train()

if __name__ == '__main__':
	main()