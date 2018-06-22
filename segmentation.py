#Nome: Lucas Yudi Sugi 						Numero USP: 9293251
#Nome: Ricardo França Fernandes do Vale 	Numero USP: 9293477
#SCC 0251 - Processamento de Imagens - 2018/1o sem - Prof. Moacir Ponti
#Aplicando as tecnicas de segmentacao nas imagens do dataset
# -*- coding: utf-8 -*-

import numpy as np

#Size of face
sizeFace = 101

#Size of subImage
sizeSubImage = 25

threshold = 200

#Extract a sub image
def extractImage(img,size):
    
    #dimensions of image
    M = img.shape[0]
    N = img.shape[1]
    
    a = int((size-1)/2)
    b = int((size-1)/2)

    #Extract the sub image
    return np.copy(img[int((M/2))-a:int((M/2))+a+1,int((N/2))-b:int((N/2))+b+1])

#Populating the mask
def populateMask(mask,size,face):

    #dimensions of image
    M = mask.shape[0]
    N = mask.shape[1]
    
    a = int((size-1)/2)
    b = int((size-1)/2)
   
    #Setting mask
    mask[int((M/2))-a:int((M/2))+a+1,int((N/2))-b:int((N/2))+b+1] = face

    return mask

#Extract the mask that select the face
def extractMask(img,face,mean):
    
    #mask
    mask = np.zeros([img.shape[0],img.shape[1],3], dtype=np.uint8)

    #Growing the region that is close to the mean
    M = face.shape[0]
    N = face.shape[1]
    
    #Creating the mask
    for i in range(M):
        for j in range(N):
            colorR = np.absolute(face[i][j][0]-mean[0])
            colorG = np.absolute(face[i][j][1]-mean[1])
            colorB = np.absolute(face[i][j][2]-mean[2])
            color = np.sum(colorR+colorG+colorB)
            if(color < threshold):
                face[i][j] = 1
            else:
                face[i][j] = 0 

    return populateMask(mask,sizeFace,face)

#Apply the segmentation
def segmentation(img):
    
    #Face of person
    face = extractImage(img,sizeFace)
    
    #subImage that will be used for computed the mean colors
    subImage = extractImage(face,sizeSubImage)
    
    #mean RGB
    meanR = np.mean(subImage[:,:,0])
    meanG = np.mean(subImage[:,:,1])
    meanB = np.mean(subImage[:,:,2])
    
    #mask
    return extractMask(img,face,[meanR,meanG,meanB])
