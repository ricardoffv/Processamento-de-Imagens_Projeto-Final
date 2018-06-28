#Nome: Lucas Yudi Sugi 						Numero USP: 9293251
#Nome: Ricardo França Fernandes do Vale 	Numero USP: 9293477
#SCC 0251 - Processamento de Imagens - 2018/1o sem - Prof. Moacir Ponti

import numpy as np
import imageio
import filter
import resize
import segmentation
import descriptors
import time 

#Read the dataset image
def readImage(nameImage):
    
    #Read the image
    img = imageio.imread(nameImage)

    return img

#Reading images to populate dataset
images = open('./images.txt', 'r')
files = images.read().splitlines()

#Creating dataset file
dataset = open('./dataset.txt', 'x')

'''
Dataset classes:
Each 20 images belong to differente classes, i.e., football team functions
The class will be determined by int(index/20), so the classes are
0 - Center Forward (Atacante)
1 - Goalkeeper (Goleiro)
2 - Side Back (Lateral)
3 - Midfielder (Meia)
4 - Wing Forward (Ponta)
5 - Center Back (Zagueiro)
'''

#Applying techniques to generate description values (for texture and gradient)
#Same process in main.py
start = time.time()
for i in range(len(files)):
	img = readImage(files[i])

	#temp image
	tmpImage = np.copy(img)

	#Pre-processing - Improving the image
	tmpImage = filter.smoothing(tmpImage)
	tmpImage = filter.histogramEqualizing(tmpImage)

	#Pre-processing - Edge enhancement
	tmpImage = filter.laplacianOfGaussian(tmpImage)

	#Resizes image
	tmpImage = resize.resize(tmpImage)
	img = resize.resize(img)

	#Segmentation image
	tmpImage = segmentation.segmentation(tmpImage)

	#Multiplying to get segmented image
	img = np.multiply(tmpImage, img)

	#Converting image to greyscale to get descriptors from image
	imgGrey = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
	imgGrey[:,:] =  np.uint8(np.floor(0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]))

	#Getting descriptors from image
	haralick = descriptors.texture_descriptors(imgGrey)
	angleHistogram = descriptors.gradient_descriptors(imgGrey)
	imgDescriptors = np.concatenate((haralick, angleHistogram), axis=0)
	
	#Creating dataset line with the description features and the player position class
	datasetLine = np.concatenate((imgDescriptors, np.array([int(i/20)])), axis=0)

	#Populating dataset, writing on file
	print(files[i])
	print(datasetLine)
	dataset.write(str(datasetLine)+'\n')
end = time.time()

print("Processo de criacao do dataset durou "+str(end-start)+"s!")













