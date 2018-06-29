'''
Authors
Lucas Yudi Sugi 				            Numero USP: 9293251
Ricardo França Fernandes do Vale 	        Numero USP: 9293477

Discipline
SCC 0251 - Processamento de Imagens - 2018/1o sem - Prof. Moacir Ponti

Title
Apply the segmentation techniques
'''

# -*- coding: utf-8 -*-

import numpy as np

#Size of face
sizeFace = 101

#Size of subImage
sizeSubImage = 25

#Threshold used to compute the mask
threshold = 200

'''
Extract a sub image
img: Image that we want to extract a subImage
size: Size of subImage
'''
def extractImage(img,size):
    
    #Dimensions of image
    M = img.shape[0]
    N = img.shape[1]
    
    #Representing the region that we want
    a = int((size-1)/2)
    b = int((size-1)/2)

    #Extract the sub image
    return np.copy(img[int((M/2))-a:int((M/2))+a+1,int((N/2))-b:int((N/2))+b+1])

'''
Populating the mask
mask: Mask that we want to populate
size: Size of face
face: Face that we extract from image
'''
def populateMask(mask,size,face):

    #dimensions of image
    M = mask.shape[0]
    N = mask.shape[1]
    
    a = int((size-1)/2)
    b = int((size-1)/2)
   
    #Copy the pixels from face to mask that is the same size of original image
    mask[int((M/2))-a:int((M/2))+a+1,int((N/2))-b:int((N/2))+b+1] = face

    return mask

'''
Extract the mask that select the face
img: Original image from segmentation
face: Face of person
mean: RGB values that is the mean of one region
'''
def computesMask(img,face,mean):
    
    #Create the matrix that is the mask
    mask = np.zeros([img.shape[0],img.shape[1],3], dtype=np.uint8)

    #Dimensions of image
    M = face.shape[0]
    N = face.shape[1]
    
    #Creating the mask
    for i in range(M):
        for j in range(N):
            #Distance of the pixel and the mean
            colorR = np.absolute(face[i][j][0]-mean[0])
            colorG = np.absolute(face[i][j][1]-mean[1])
            colorB = np.absolute(face[i][j][2]-mean[2])
            color = np.sum(colorR+colorG+colorB)

            #Threshold of image
            if(color < threshold):
                face[i][j] = 1
            else:
                face[i][j] = 0 

    return populateMask(mask,sizeFace,face)

'''
Apply the segmentation
img: Imagem that will be segmented
'''
def segmentation(img):
    
    #Extract one submatrix that has the face of person
    face = extractImage(img,sizeFace)
    
    #Extract one submatrix that will be used for compute the mean color
    subImage = extractImage(face,sizeSubImage)
    
    #Calculates the mean color for channel RGB
    meanR = np.mean(subImage[:,:,0])
    meanG = np.mean(subImage[:,:,1])
    meanB = np.mean(subImage[:,:,2])
    
    #Extract and return the mask
    return computesMask(img,face,[meanR,meanG,meanB])
