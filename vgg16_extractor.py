import numpy as np
import tensorflow as tf
import os
import json
import pandas  as pd
import pickle
import time
from scipy import ndimage

from vgg.vgg16 import *

def _process_caption_data(caption_file, image_dir, max_length):
	with open(caption_file) as f:
		caption_data = json.load(f)

	# id_to_filename is a dictionary such as {image_id: filename]} 
	id_to_filename = {image['id']: image['file_name'] for image in caption_data['images']}

	# with open("test.json", "w+") as fp:
	#     (json.dump(caption_data, fp, indent=4, sort_keys=True))

	# print(len(id_to_filename.keys()))

	# data is a list of dictionary which contains 'captions', 'file_name' and 'image_id' as key.
	data = []
	for annotation in caption_data['annotations']:
		image_id = annotation['image_id']
		annotation['file_name'] = os.path.join(image_dir, id_to_filename[image_id])
		data += [annotation]

	print("HERE")

	# with open("data.json", "w+") as fp:
	#     (json.dump(data, fp, indent=4, sort_keys=True))

	# convert to pandas dataframe (for later visualization or debugging)
	caption_data = pd.DataFrame.from_dict(data)
	del caption_data['id']
	caption_data.sort_values(by='image_id', inplace=True)
	caption_data = caption_data.reset_index(drop=True)

	del_idx = []
	for i, caption in enumerate(caption_data['caption']):
		caption = caption.replace('.','').replace(',','').replace("'","").replace('"','')
		caption = caption.replace('&','and').replace('(','').replace(")","").replace('-',' ')
		caption = " ".join(caption.split())  # replace multiple spaces

		caption_data.set_value(i, 'caption', caption.lower())
		if len(caption.split(" ")) > max_length:
			del_idx.append(i)

	# delete captions if size is larger than max_length
	print("The number of captions before deletion: %d" %len(caption_data))
	caption_data = caption_data.drop(caption_data.index[del_idx])
	caption_data = caption_data.reset_index(drop=True)

	cd = caption_data.to_dict('dict')

	# print(len(cd[cd.keys()[0]]))
	# print(len(cd['caption']))
	# print(len(cd['file_name']))
	# print(len(cd['caption']))

	newData = []
	for i in range(0, len(cd['caption'])):
		temp = {}
		temp['caption'] = cd['caption'][i]
		temp['fileName'] = cd['file_name'][i]
		# print(cd['image_id'][i].item())
		# input()
		temp['imageId'] = cd['image_id'][i].item()
		newData += [temp]        

	newData = sorted(newData, key=lambda k: k['imageId'])

	# with open("data3.json", "w+") as fp:
	#     json.dump(newData, fp, indent=4, sort_keys=True)

	print("The number of captions after deletion: %d" %len(caption_data))
	return caption_data

def load_pickle(path):
	with open(path, 'rb') as f:
		file = pickle.load(f)
		print(('Loaded %s..' %path))
		return file  

def save_pickle(data, path):
	with open(path, 'wb') as f:
		pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
		print(('Saved %s..' %path))

def main():
	batch_size = 50
	max_length = 15
	miniBatch = 1600	

	caption_file = 'data/annotations/captions_train2014.json'
	image_dir = 'image/%2014_resized/'

	# about 80000 images and 400000 captions for train dataset
	train_dataset = _process_caption_data(caption_file='data/annotations/captions_train2014.json',
												image_dir='image/train2014_resized/',
												max_length=max_length)

	save_pickle(train_dataset, 'data/train/train.annotations2.pkl')

	vgg = Vgg16()
	with tf.name_scope("content_vgg"):
		vgg.build()
	with tf.Session() as sess:		
		annotationsPath = './data/train/train.annotations2.pkl'

		# save_path = './data/train/features (vgg16)/train.features.pkl' % (split, split)

		annotations = load_pickle(annotationsPath)

		image_path = list(annotations['file_name'].unique())

		n_examples = len(image_path)

		nIters = int(np.ceil(float(n_examples)/n_examples))

		all_feats = np.ndarray([n_examples, 4096], dtype=np.float32)

		save_path = './data/train/features (vgg16)/train.features.pkl'

		print("==================================================")
		print("No of examples: ", n_examples)
		print("nIters: ", nIters)
		print("==================================================")

		f = 1

		t = time.time()

		for start, end in zip(range(0, n_examples, batch_size),
								range(batch_size, n_examples + batch_size, batch_size)):

			# print(start, end, n_examples, batch_size)
			# print(range(0, n_examples, batch_size))
			# print(range(batch_size, n_examples + batch_size, batch_size))
			# input()

			image_batch_file = image_path[start:end]

			# print(image_batch_file)
			# input()

			# continue
			# print(image_path)
			# print(image_batch_file)
			# input()
			image_batch = np.array(list(map(lambda x: ndimage.imread(x, mode='RGB'), image_batch_file))).astype(np.float32)
			feats = sess.run(vgg.fc7, feed_dict={vgg.images: image_batch})
			all_feats[start:end, :] = feats
			print(("Processed %d features.." % (end)))
			print("Time take: ", time.time() - t)
		# use hickle to save huge feature vectors
		save_pickle(all_feats, save_path)
		print(("Saved %s.." % (save_path)))

		# for s, e in zip(range(0, n_examples, miniBatch), range(miniBatch, n_examples + miniBatch, miniBatch)):
		# 	save_path = './data/train/features (vgg16)/train.features.%d.pkl' % (f)
		# 	# print("")

		# 	print("==================================================")
		# 	print(s, e, n_examples, miniBatch)

		# 	all_feats = np.ndarray([(e-s), 4096], dtype=np.float32)

		# 	print(all_feats.shape)
		# 	print("==================================================")
		# 	# print("")	

		# 	i = 0
		# 	j = batch_size
		# 	for start, end in zip(range(s, e, batch_size), range(s + batch_size, e + batch_size, batch_size)):

		# 		# print(start, end)
		# 		# images = tf.placeholder(tf.float32, [None,  224, 224, 3])

		# 		image_batch_file = image_path[start:end]
				
		# 		temp2 = map(lambda x: ndimage.imread(x, mode='RGB'), image_batch_file)
		# 		# print(type(temp2))
		# 		temp3 = list(temp2)
		# 		# print(type(temp3))
		# 		image_batch = np.array(temp3).astype(np.float32)

		# 		# feats = np.empty((0, 4096))
		# 		# for im in image_batch:
		# 		# 	feed_dict = feed_dict={images: im.reshape((1, 224, 224, 3))}
		# 		# 	vgg = Vgg16()
		# 		# 	with tf.name_scope("content_vgg"):
		# 		# 		vgg.build(images)
					
		# 		# 	temp_feats = sess.run(vgg.fc7, feed_dict=feed_dict)
		# 		# 	print(temp_feats.shape)
		# 		# 	feats = np.append(feats, temp_feats, axis=0)
		# 		# 	print(feats.shape)
		# 		# 	print("Time taken: ", time.time() - t)


		# 		# print(len(image_batch_file))
		# 		# print(len(image_path))
		# 		# print(image_batch.shape)
		# 		# input()
		# 		# image_batch = np.empty((0, 224, 224, 3))

		# 		# for im in image_batch_file:
		# 		# 	temp = load_image(im)
		# 		# 	print(temp.shape)
		# 		# 	temp = temp.reshape((1, 224, 224, 3))
		# 		# 	print(temp.shape)
		# 		# 	# input()
		# 		# 	image_batch = np.append(image_batch, temp, axis=0)
		# 		# 	print(image_batch.shape)

		# 		# input()
		# 		feed_dict = {vgg.images: image_batch}

		# 		# # input()

		# 		# # continue


		# 		feats = sess.run(vgg.fc7, feed_dict=feed_dict)

		# 		print(feats.shape)
		# 		# input()

		# 		if (j > n_examples):
		# 			j = n_examples % batch_size
		# 		all_feats[i:j, :] = feats
		# 		print(("Processed %d train features.." % (end)))

		# 		i = i + batch_size
		# 		j = j + batch_size

		# 	save_pickle(all_feats, save_path)
		# 	print(("Saved %s.." % (save_path)))
		# 	print("Time taken: ", time.time()-t)
		# 	f = f + 1 
	


if __name__ == '__main__':
	main()