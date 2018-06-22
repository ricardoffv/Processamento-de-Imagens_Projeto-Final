# -*- coding: utf-8 -*-

'''
Author: Lucas Yudi Sugi - 9293251
Discipline: SCC0251_1Sem_2018 
'''

import numpy as np
import imageio

#Size of mean filter applied in smoothing
sizeMeanFilter = 3
#a medida que aumenta o tamanho, confunde regioes e nao permite segmentacao correta

#Size of sobel filter applied in sharpening
sizeSobelFilter = 3

#Size of log filter applied in sharpening
sizeLogFilter = 3

#Sigma of Log
sigmaLog = 1.6

#parameter that is used in gamma adjustment
gamma = 0.7

#parameter that us used in highBoost
highBoostParameter = 2

#Improve the enhancement of image using histogram equalizing
def histogramEqualizing(img):

    #Max pixel in image
    maxPixel = 256;
    
    #creating the histogram
    histR = np.zeros(maxPixel,int)
    histG = np.zeros(maxPixel,int)
    histB = np.zeros(maxPixel,int)
    
    #populating the histogram
    for i in range(maxPixel):
            histR[i] =  (img[:,:,0] == i).sum()
            histG[i] =  (img[:,:,1] == i).sum()
            histB[i] =  (img[:,:,2] == i).sum()
    
    #accumulating pixels
    for i in range(1,maxPixel):
        histR[i] = histR[i] + histR[i-1]
        histG[i] = histG[i] + histG[i-1]
        histB[i] = histB[i] + histB[i-1]
    
    #Dimension of image
    M = img.shape[0]
    N = img.shape[1]
    
    #multiplicative factor
    mulFactor = (maxPixel-1)/(M*N)
    
    #new image
    newImg = np.zeros((M,N,3), dtype=np.uint8)
    
    #Apply the histogram equalizer
    for i in range(maxPixel):
        sR = int(histR[i] * mulFactor)
        sG = int(histG[i] * mulFactor)
        sB = int(histB[i] * mulFactor)

        newImg[np.where(img[:,:,0]==i)] = sR
        newImg[np.where(img[:,:,1]==i)] = sG
        newImg[np.where(img[:,:,2]==i)] = sB
    
    return newImg

#Improve the enhancement of image using gamma adjustment
def gammaAdjustment(img):
    
    img = np.power(img,gamma)
    img = np.uint8(((img-img.max())/(img.max()-img.min()))*255)

    return img

#Apply the convolution
def convolution(img,mask):
    
    #Extrac color channel
    imgR = img[:,:,0]
    imgG = img[:,:,1]
    imgB = img[:,:,2]

    #image in frequency domain
    imgR = np.fft.fft2(imgR)
    imgG = np.fft.fft2(imgG)
    imgB = np.fft.fft2(imgB)
   
    #filter in frequency domain
    mask = np.fft.fft2(mask)

    #apply the convolution
    imgR = np.multiply(mask,imgR)
    imgG = np.multiply(mask,imgG)
    imgB = np.multiply(mask,imgB)
   
    #real part
    imgR = np.real(np.fft.ifft2(imgR))
    imgG = np.real(np.fft.ifft2(imgG))
    imgB = np.real(np.fft.ifft2(imgB))

    #normalizing
    imgR = np.uint8(((imgR-imgR.min())/(imgR.max()-imgR.min()))*255)
    imgG = np.uint8(((imgG-imgG.min())/(imgG.max()-imgG.min()))*255)
    imgB = np.uint8(((imgB-imgB.min())/(imgB.max()-imgB.min()))*255)
    
    #image with result
    imgResult = np.zeros([img.shape[0],img.shape[1],3],dtype=np.uint8)
    imgResult[:,:,0] = imgR
    imgResult[:,:,1] = imgG
    imgResult[:,:,2] = imgB

    return imgResult

#Smoothing the image with mean filter
def smoothing(img):

    #Dimension of image
    M = img.shape[0]
    N = img.shape[1]

    #creating mean filter in domain frequency
    meanFilter = np.zeros([M,N])
    for i in range(sizeMeanFilter):
        for j in range(sizeMeanFilter):
            meanFilter[i][j] = 1/(sizeMeanFilter*sizeMeanFilter)
    
    return convolution(img,meanFilter)

#Apply the equation of log
def log(x,y):
    return (-1/(np.pi*np.power(sigmaLog,4))) * (1-((np.power(x,2)+np.power(y,2))/(2*np.power(sigmaLog,2))))* (np.exp((-np.power(x,2)-np.power(y,2))/(2*np.power(sigmaLog,2)))) 

#Sharpening with laplacian of gaussian
def laplacianOfGaussian(img):
    
    #Dimension of image
    M = img.shape[0]
    N = img.shape[1]
    
    #create filter
    laplacianFilter = np.zeros([M,N])
    a = (sizeLogFilter-1/2)
    b = (sizeLogFilter-1/2)
    for i in range(sizeLogFilter):
        for j in range(sizeLogFilter):
            laplacianFilter[i][j] = log(i-a,j-b)  

    return convolution(img,laplacianFilter)

#Sharpening with high boost
def highBoost(img):

    #Blur image
    tempImage = np.copy(img)
    tempImage= smoothing(tempImage)
    
    #Create mask
    mask = img-tempImage

    return np.uint8(highBoostParameter*mask)

#Sharpening with sobel operator
def sobel(img): 
    
    #Dimension of image
    M = img.shape[0]
    N = img.shape[1]
    
    #Create the two filters
    Fx = np.zeros([M,N]) 
    Fx[0][0] = Fx[2][0] = 1
    Fx[1][0] = 2
    Fx[0][2] = Fx[2][2] = -1
    Fx[1][2] = -2
    Fy = np.zeros([M,N])
    Fy[0:3,0:3] = np.transpose(Fx[0:3,0:3])
    
    #Apply convolution
    Fx = convolution(img,Fx)
    Fy = convolution(img,Fy)

    #convert
    Fx = np.float64(Fx)
    Fy = np.float64(Fy)
    
    return np.uint8(np.sqrt(np.power(Fx,2) + np.power(Fy,2)))
