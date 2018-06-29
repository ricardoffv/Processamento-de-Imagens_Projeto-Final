'''
Authors
Lucas Yudi Sugi 							Numero USP: 9293251
Ricardo França Fernandes do Vale 	        Numero USP: 9293477
Discipline
SCC 0251 - Processamento de Imagens - 2018/1o sem - Prof. Moacir Ponti

Title
DWNN and KNN classification algorithm to find the class of the input picture
'''

import numpy as np

#Setting the dataset file and the default size of features and class, alongside the number of instances
data = "./dataset.txt"
lineSize = 24
datasetSize = 120

#ML algorithm parameter with value set default
sigma = 0.5

'''
Generating features and classes for the 120 instances of the dataset
'''
def generateDataset():

	#Reading dataset
	images = open(data, 'r')
	values = images.read().split()

	#Setting features vector for all images and their classes also 
	features = np.zeros([datasetSize, lineSize-1], dtype=float) 
	classes = np.zeros(datasetSize, dtype=float) 

	#Populating the arrays
	counter = -1
	for i in range(len(values)):
		
		index = i%lineSize
		if (index == 0):
			counter += 1
		
		if (index != lineSize-1):
			features[counter, index] = float(values[i])
		else:
			classes[counter] = float(values[i])

	return [features, classes]

'''
Applying the DWNN classification algorithm, the pourpose is returning the class receiving a set of input features
inputFeatures: input image descriptors 
'''
def dwnn(inputFeatures):

	#Getting the features and the classes of the instances 
	[features, classes] = generateDataset()

	#Calculating the algorithm weights
	weights = np.zeros(features.shape[0], dtype=float)
	for i in range(features.shape[0]):
		dist = np.sqrt(np.sum((inputFeatures - features[i])**2))
		weights[i] = np.exp(-(dist**2))/(2*(sigma**2))

	#Extracting predicted class
	tag = np.sum(np.multiply(weights,classes))/np.sum(weights) 

	return tag

'''
Applying the 1NN classification algorithm, the pourpose is returning the class receiving a set of input features
inputFeatures: input image descriptors 
'''
def oneNN_nearest_neighbor(inputFeatures):

	#Getting the features and the classes of the instances 
	[features, classes] = generateDataset()

	#Calculating the distances
	dist = np.zeros(features.shape[0], dtype=float)
	for i in range(features.shape[0]):
		dist[i] = np.sqrt(np.sum((inputFeatures - features[i])**2)) 

	#Extracting index from the distance vector which has the lower value 
	index = np.nanargmin(dist)

	#Return of the lower distance class
	return classes[index]