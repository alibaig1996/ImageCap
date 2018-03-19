import pickle
import time
import pandas
import os

val2014 = [f for f in os.listdir("D:\ImageCap\image\\val2014_resized")]
val2017 = [f for f in os.listdir("D:\ImageCap\image\\val2017")]
print(len(val2014))
print(len(val2017))

print(val2014[0].replace('COCO_val2014_', ''))
print(val2017[0])
# quit()
# with open('train.annotations.pkl', 'rb') as f:
#     data1 = pickle.load(f).to_dict('dict')
#     # print(type(data))
#     newData1 = []
#     for i in range(0, len(data1['caption'])):
#         temp = {}
#         temp['caption'] = data1['caption'][i]
#         temp['fileName'] = data1['file_name'][i]
#         # print(data1['image_id'][i].item())
#         # input()
#         temp['imageId'] = data1['image_id'][i].item()
#         newData1 += [temp]        

#     newData1 = sorted(newData1, key=lambda k: k['imageId'])
#     print(len(newData1))

# with open('train.annotations2.pkl', 'rb') as f:
#     data2 = pickle.load(f).to_dict('dict')
#     # print(type(data))
#     newData2 = []
#     for i in range(0, len(data2['caption'])):
#         temp = {}
#         temp['caption'] = data2['caption'][i]
#         temp['fileName'] = data2['file_name'][i]
#         # print(data2['image_id'][i].item())
#         # input()
#         temp['imageId'] = data2['image_id'][i].item()
#         newData2 += [temp]        

#     newData2 = sorted(newData2, key=lambda k: k['imageId'])
#     print(len(newData2))
#     # print(type(data2))

# with open('word_to_idx.pkl', 'rb') as f:
#     data = pickle.load(f)
#     print(type(data))
#     print(len(data))

# with open('word_to_idx2.pkl', 'rb') as f:
#     data = pickle.load(f)
#     print(type(data))
#     print(len(data))

# data1 = data1.tolist()
# data2 = data2.tolist() 

count = 0

t = time.time()

for i in range(0, len(val2014)):
	for j in range(0, len(val2017)):
		# print(data1[i].replace('image/train2014_resized/COCO_train2014_', ''))
		# print(data2[j].replace('image/train2017/', ''))
		# input()
		if (val2014[i].replace('COCO_val2014_', '') == val2017[j]):
			# print("HERE")
			val2017.remove(val2017[j])
			count = count + 1
			break

	if (i % 5000 == 0):
		print("Processed ", i, " images! Time taken: ", time.time() - t)

# # print(data1[0].replace('image/train2014_resized/COCO_train2014_', ''))
# # print(data2[1].replace('image/train2017/', ''))

print(count)