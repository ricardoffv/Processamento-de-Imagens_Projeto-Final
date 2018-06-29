'''
Authors
Lucas Yudi Sugi 							Numero USP: 9293251
Ricardo França Fernandes do Vale 	        Numero USP: 9293477
Discipline
SCC 0251 - Processamento de Imagens - 2018/1o sem - Prof. Moacir Ponti

Title
Main project
'''

import numpy as np
import imageio
import sys
import matplotlib.pyplot as plt
import filter
import resize
import segmentation
import descriptors
import classification

#Read the input image
def readImage():
    
    #Read the name of image
    nameImage = sys.argv[1]
    
    #Read the image
    img = imageio.imread(nameImage)

    return img

#Called for function that reads the image
img = readImage()

#temp image that will be used for extract a mask
tmpImage = np.copy(img)

#Pre-processing - Improving the image
tmpImage = filter.smoothing(tmpImage)
tmpImage = filter.histogramEqualizing(tmpImage)

#Pre-processing - Edge enhancement
tmpImage = filter.laplacianOfGaussian(tmpImage);

#Resizes image
tmpImage = resize.resize(tmpImage)
img = resize.resize(img)

#Segmentation image
tmpImage = segmentation.segmentation(tmpImage);

#Multiplying to get segmented image
img = np.multiply(tmpImage, img)

#Converting image to greyscale to get descriptors from image
imgGrey = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
imgGrey[:,:] =  np.uint8(np.floor(0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]))

#Getting descriptors from image
haralick = descriptors.texture_descriptors(imgGrey)
angleHistogram = descriptors.gradient_descriptors(imgGrey)

#Creating input features vector with the description features
inputFeatures = np.concatenate((haralick, angleHistogram), axis=0)

#Applying classification algorithm to return image class
tag = classification.oneNN_nearest_neighbor(inputFeatures)

#Showing result
if (tag == 0.0):
	print("Parabéns! Você é um ATACANTE!")
elif (tag == 1.0):
	print("Parabéns! Você é um GOLEIRO!")
elif (tag == 2.0):
	print("Parabéns! Você é um LATERAL!")
elif (tag == 3.0):
	print("Parabéns! Você é um MEIA!")
elif (tag == 4.0):
	print("Parabéns! Você é um PONTA!")
elif (tag == 5.0):
	print("Parabéns! Você é um ZAGUEIRO!")

#Write image
imageio.imwrite('out.jpg',img)
