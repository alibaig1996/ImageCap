import numpy as np
import tensorflow as tf

import vgg16
import utils

img1 = utils.load_image("./test_data/tiger.jpeg")
print((img1.shape))
batch1 = img1.reshape((1, 224, 224, 3))
print(batch1.shape)
img2 = utils.load_image("./test_data/test.jpg")

batch2 = img2.reshape((1, 224, 224, 3))
x = np.empty((0, 224, 224, 3))
batch = np.append(x, batch2, axis=0)
print(batch.shape)
batch = np.concatenate((batch, batch1), 0).astype(np.float32)
batch = np.concatenate((batch, batch1), 0).astype(np.float32)
batch = np.concatenate((batch, batch1), 0).astype(np.float32)
batch = np.concatenate((batch, batch1), 0).astype(np.float32)
batch = np.concatenate((batch, batch1), 0).astype(np.float32)
batch = np.concatenate((batch, batch1), 0).astype(np.float32)
batch = np.concatenate((batch, batch1), 0).astype(np.float32)
batch = np.concatenate((batch, batch1), 0).astype(np.float32)
batch = np.concatenate((batch, batch1), 0).astype(np.float32)
    


print(batch.shape)
# img1 = utils.load_image("./test_data/tiger.jpeg")
# batch1 = img1.reshape((1, 224, 224, 3))
# batch = np.concatenate((batch, batch1), 0)
# print(batch.shape)

# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
# with tf.device('/cpu:0'):
vgg = vgg16.Vgg16()
with tf.Session() as sess:
    for i in range(0, 5):
        images = tf.placeholder("float", [None, 224, 224, 3])
        feed_dict = {images: batch}

        with tf.name_scope("content_vgg"):
            vgg.build(images)

        prob = sess.run(vgg.fc7, feed_dict=feed_dict)
        print(prob.shape)
        # conv5_3 = sess.run(vgg.conv5_3, feed_dict=feed_dict)
        # print("CONV5_3: ")
        # print(sess.run(vgg.conv5_3, feed_dict=feed_dict).shape)
        # print("========================================")
        # print("POOL5: ")
        # print(sess.run(vgg.pool5, feed_dict=feed_dict).shape)
        # print("========================================")
        # print("POOL5: ")
        # print(sess.run(vgg.pool5, feed_dict=feed_dict).shape)
        # print("========================================")
        # print("FC6: ")
        # print(sess.run(vgg.fc6, feed_dict=feed_dict).shape)
        # print("========================================")
        # print("RELU6: ")
        # print(sess.run(vgg.relu6, feed_dict=feed_dict).shape)
        # print("========================================")
        # print("FC7: ")
        # print(sess.run(vgg.fc7, feed_dict=feed_dict).shape)
        # print("========================================")
        # print("RELU7: ")
        # print(sess.run(vgg.relu7, feed_dict=feed_dict).shape)
        # print("========================================")
        # print("FC8: ")
        # print(sess.run(vgg.fc8, feed_dict=feed_dict).shape)
        # print("========================================")
        # utils.print_prob(prob[0], './synset.txt')
        # utils.print_prob(prob[1], './synset.txt')