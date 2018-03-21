import numpy as np
import tensorflow as tf
import os
import json
import pandas  as pd
import pickle
import time
from scipy import ndimage
from collections import Counter
from vgg.vgg16 import *


def processCaptionData(caption_file, image_dir, max_length):
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

def buildVocab(annotations, threshold=1):
    counter = Counter()
    max_len = 0

    # print(annotations['caption'])
    # input()

    for i, caption in enumerate(annotations['caption']):
        # print(caption)
        # input()
        words = caption.split(' ') # caption contrains only lower-case words
        for w in words:
            counter[w] +=1
        
        if len(caption.split(" ")) > max_len:
            max_len = len(caption.split(" "))

    vocab = [word for word in counter if counter[word] >= threshold]
    print(('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold)))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx = 3
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    print("Max length of caption: ", max_len)
    return word_to_idx

def buildCaptionVector(annotations, word_to_idx, max_length=15):
    n_examples = len(annotations)
    # n_examples = 1000    
    captions = np.ndarray((n_examples,max_length+2)).astype(np.int32)   


    for i, caption in enumerate(annotations['caption']):
        # if (i == 1000):
        #     break
        words = caption.split(" ") # caption contrains only lower-case words
        cap_vec = []
        cap_vec.append(word_to_idx['<START>'])
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
        cap_vec.append(word_to_idx['<END>'])
        
        # pad short caption with the special null token '<NULL>' to make it fixed-size vector
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>']) 
        
        captions[i, :] = np.asarray(cap_vec)

    # print(captions)
    print("Finished building caption vectors")
    return captions

# def oneHotEncode(word, word_to_idx):
# 	vec = np.zeros(len(word_to_idx))

# 	index = word_to_idx[word]

# 	vec[index] = 1

# 	return vec

# def buildCaptionVector(annotations, word_to_idx, max_length=15):
#     n_examples = len(annotations)
#     # n_examples = 1000    
#     captions = np.ndarray((n_examples, max_length+2, len(word_to_idx))).astype(np.int32) 
#     print(captions.nbytes)
#     quit()
#     for i, caption in enumerate(annotations['caption']):
#         # if (i == 1000):
#         #     break
#         words = caption.split(" ") # caption contrains only lower-case words
#         cap_vec = []
#         cap_vec.append(oneHotEncode('<START>', word_to_idx))
#         for word in words:
#             if word in word_to_idx:
#                 cap_vec.append(oneHotEncode(word, word_to_idx))
#         cap_vec.append(oneHotEncode('<END>', word_to_idx))
        
#         # pad short caption with the special null token '<NULL>' to make it fixed-size vector
#         if len(cap_vec) < (max_length + 2):
#             for j in range(max_length + 2 - len(cap_vec)):
#                 cap_vec.append(oneHotEncode('<NULL>', word_to_idx))

#         print(word, ": ", cap_vec) 

#         captions[i, :] = np.asarray(cap_vec)

#     # print(captions)
#     print("Finished building caption vectors")
    # return captions

def buildFileNames(annotations, split):
    image_file_names = []
    id_to_idx = {}
    id_to_idx2 = {}
    idx = 0
    image_ids = annotations['image_id']
    file_names = annotations['file_name']
    for image_id, file_name in zip(image_ids, file_names):
        if not image_id in id_to_idx:
            id_to_idx[image_id] = idx
            id_to_idx2[str(image_id)] = str(idx)
            image_file_names.append(file_name)
            idx += 1

    # if (split == 'val'):
    #     print(len(id_to_idx))
    #     with open("data4.json", "w+") as fp:
    #         json.dump(id_to_idx2, fp, indent=4, sort_keys=True)
        
    # id_to_idx2 = sorted(id_to_idx2, key=lambda k: k['image_id'])
    # quit()
    # id_to_idx = sorted(id_to_idx, key=lambda k: k['image_id'])
    # print(len(image_file_names))
    # print(type(image_file_names[0]))
    # print(type(id_to_idx))
    # for i in id_to_idx.keys():
    #     print("")
    #     print(i)
    #     print(id_to_idx[i])
    #     input()
    #     # break
    # input()

    file_names = np.asarray(image_file_names)
    # print(file_names)
    return file_names, id_to_idx

def buildImageIdx(annotations, id_to_idx, split):
    image_idxs = np.ndarray(len(annotations), dtype=np.int32)
    # image_idxs = np.ndarray(1000, dtype=np.int32)

    if (split == 'val'):
        print(image_idxs.shape)
        print("")

    image_ids = annotations['image_id']
    # print(type(image_ids))
    # print("")
    # print(image_ids[0])
    # print(image_ids[1])
    # print(image_ids[2])
    # print(image_ids[3])
    # print("")

    for i, image_id in enumerate(image_ids):
        # if (i == 1000):
        #     break

        image_idxs[i] = id_to_idx[image_id]
        
        # if(split  == 'val'):
            # print(i, image_id)
            # print(i)
            # print(image_id)
            # print(id_to_idx[image_id])
            # print(image_idxs[i])
            # print("")
            # input() 

    return image_idxs

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
    # batch size for extracting feature vectors from vggnet.
    batch_size = 50
    # maximum length of caption(number of word). if caption is longer than max_length, deleted.  
    max_length = 25
    # if word occurs less than word_count_threshold in training dataset, the word index is special unknown token.
    word_count_threshold = 1

    caption_file = 'data/annotations/captions_train2014.json'
    image_dir = 'image/%2014_resized/'

    # about 80000 images and 400000 captions for train dataset
    train_dataset = processCaptionData(caption_file='data/annotations/captions_train2017.json',
                                          image_dir='image/train_resized/',
                                          max_length=max_length)

    # about 40000 images and 200000 captions
    val_dataset = processCaptionData(caption_file='data/annotations/captions_val2014.json',
                                        image_dir='image/val_resized/',
                                        max_length=max_length)

    # about 4000 images and 20000 captions for val / test dataset
    val_cutoff = int(0.1 * len(val_dataset))
    test_cutoff = int(0.2 * len(val_dataset))
    print('Finished processing caption data')

    save_pickle(train_dataset, 'data/train/train.annotations.pkl')
    save_pickle(val_dataset[:val_cutoff], 'data/val/val.annotations.pkl')
    save_pickle(val_dataset[val_cutoff:test_cutoff].reset_index(drop=True), 'data/test/test.annotations.pkl')

    for split in ['train', 'val', 'test']:
        annotations = load_pickle('./data/%s/%s.annotations.pkl' % (split, split))

        if split == 'train':
            word_to_idx = buildVocab(annotations=annotations, threshold=word_count_threshold)
            save_pickle(word_to_idx, './data/%s/word_to_idx.pkl' % split)
        
        captions = buildCaptionVector(annotations=annotations, word_to_idx=word_to_idx, max_length=max_length)
        save_pickle(captions, './data/%s/%s.captions.pkl' % (split, split))

        file_names, id_to_idx = buildFileNames(annotations, split)
        save_pickle(file_names, './data/%s/%s.file.names.pkl' % (split, split))

        image_idxs = buildImageIdx(annotations, id_to_idx, split)
        save_pickle(image_idxs, './data/%s/%s.image.idxs.pkl' % (split, split))

        # prepare reference captions to compute bleu scores later
        image_ids = {}
        feature_to_captions = {}
        i = -1
        for caption, image_id in zip(annotations['caption'], annotations['image_id']):
            if not image_id in image_ids:
                image_ids[image_id] = 0
                i += 1
                feature_to_captions[i] = []
            feature_to_captions[i].append(caption.lower() + ' .')
        save_pickle(feature_to_captions, './data/%s/%s.references.pkl' % (split, split))
        print("Finished building %s caption dataset" %split)

    # quit()
    # extract fc7 feature maps
    vgg = Vgg16()
    vgg.build()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for split in ['train', 'val', 'test']:
            anno_path = './data/%s/%s.annotations.pkl' % (split, split)
            save_path = './data/%s/%s.features.pkl' % (split, split)
            annotations = load_pickle(anno_path)
            image_path = list(annotations['file_name'].unique())
            n_examples = len(image_path)
                    
            all_feats = np.ndarray([n_examples, 4096], dtype=np.float32)

            print(all_feats.shape)
            print("")

            for start, end in zip(range(0, n_examples, batch_size),
                                  range(batch_size, n_examples + batch_size, batch_size)):

                image_batch_file = image_path[start:end]

                image_batch = np.array(list(map(lambda x: ndimage.imread(x, mode='RGB'), image_batch_file))).astype(np.float32)
                
                feats = sess.run(vgg.fc7, feed_dict={vgg.images: image_batch})
                
                all_feats[start:end, :] = feats
                
                print(("Processed %d %s features.." % (end, split)))

            # use hickle to save huge feature vectors
            save_pickle(all_feats, save_path)
            print(("Saved %s.." % (save_path)))


if __name__ == "__main__":
    main()