'''
Authors
Lucas Yudi Sugi 				Numero USP: 9293251
Ricardo França Fernandes do Vale 	        Numero USP: 9293477

Discipline
SCC 0251 - Processamento de Imagens - 2018/1o sem - Prof. Moacir Ponti

Title
Apply the resize in an image
'''

# -*- coding: utf-8 -*-

import numpy as np
import imageio

#Size of mean filter applied in smoothing
sizeMeanFilter = 3

#Size of sobel filter applied in sharpening
sizeSobelFilter = 3

#Size of log filter applied in sharpening
sizeLogFilter = 3

#Sigma used in Log
sigmaLog = 1.6

#Parameter that is used in gamma adjustment
gamma = 0.7

#Parameter that us used in highBoost
highBoostParameter = 2

'''
Improve the enhancement of image and convert to grayscale using histogram equalizing
img: Image that we want to calculate
'''
def histogramEqualizing(img):

    #Max pixel in image
    maxPixel = 256;
    
    #Creating the histogram for three channels
    histR = np.zeros(maxPixel,int)
    histG = np.zeros(maxPixel,int)
    histB = np.zeros(maxPixel,int)
    
    #Populating the channels
    for i in range(maxPixel):
            histR[i] =  (img[:,:,0] == i).sum()
            histG[i] =  (img[:,:,1] == i).sum()
            histB[i] =  (img[:,:,2] == i).sum()
    
    #Accumulating pixels
    for i in range(1,maxPixel):
        histR[i] = histR[i] + histR[i-1]
        histG[i] = histG[i] + histG[i-1]
        histB[i] = histB[i] + histB[i-1]
    
    #Dimension of image
    M = img.shape[0]
    N = img.shape[1]
    
    #Multiplicative factor
    mulFactor = (maxPixel-1)/(M*N)
    
    #New image that is equalized
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

'''
Improve the enhancement of image using gamma adjustment
img: Image that we want to enhancement
'''
def gammaAdjustment(img):
    
    #Computhes the gamma adjustment
    img = np.power(img,gamma)

    #Normalizing and convert
    img = np.uint8(((img-img.max())/(img.max()-img.min()))*255)

    return img

'''
Apply the convolution in an image
img: Image that it will be used for convolution
mask: Matrix with the mask that will be used for convolution
'''
def convolution(img,mask):
    
    #Extrac color channel
    imgR = img[:,:,0]
    imgG = img[:,:,1]
    imgB = img[:,:,2]

    #Image in frequency domain
    imgR = np.fft.fft2(imgR)
    imgG = np.fft.fft2(imgG)
    imgB = np.fft.fft2(imgB)
   
    #Filter in frequency domain
    mask = np.fft.fft2(mask)

    #Apply the convolution
    imgR = np.multiply(mask,imgR)
    imgG = np.multiply(mask,imgG)
    imgB = np.multiply(mask,imgB)
   
    #Get the real part
    imgR = np.real(np.fft.ifft2(imgR))
    imgG = np.real(np.fft.ifft2(imgG))
    imgB = np.real(np.fft.ifft2(imgB))

    #Normalizing
    imgR = np.uint8(((imgR-imgR.min())/(imgR.max()-imgR.min()))*255)
    imgG = np.uint8(((imgG-imgG.min())/(imgG.max()-imgG.min()))*255)
    imgB = np.uint8(((imgB-imgB.min())/(imgB.max()-imgB.min()))*255)
    
    #Image with result
    imgResult = np.zeros([img.shape[0],img.shape[1],3],dtype=np.uint8)
    imgResult[:,:,0] = imgR
    imgResult[:,:,1] = imgG
    imgResult[:,:,2] = imgB

    return imgResult

'''
Smoothing the image with mean filter
img: Image that will be smoothed
'''
def smoothing(img):

    #Dimension of image
    M = img.shape[0]
    N = img.shape[1]

    #Creating mean filter
    meanFilter = np.zeros([M,N])
    for i in range(sizeMeanFilter):
        for j in range(sizeMeanFilter):
            meanFilter[i][j] = 1/(sizeMeanFilter*sizeMeanFilter)
    
    return convolution(img,meanFilter)

'''
Apply the equation of log
x: Coordinate 'x' that we use for calculate log
y: Coordinate 'y' that we use for calculate log
'''
def log(x,y):
    return (-1/(np.pi*np.power(sigmaLog,4))) * (1-((np.power(x,2)+np.power(y,2))/(2*np.power(sigmaLog,2))))* (np.exp((-np.power(x,2)-np.power(y,2))/(2*np.power(sigmaLog,2)))) 

'''
Sharpening with laplacian of gaussian
img: Image that we use to apply log
'''
def laplacianOfGaussian(img):
    
    #Dimension of image
    M = img.shape[0]
    N = img.shape[1]
    
    #Create filter
    laplacianFilter = np.zeros([M,N])

    #Region that we want to populate (Calculates the limit's image)
    a = (sizeLogFilter-1/2)
    b = (sizeLogFilter-1/2)

    #Populating mask
    for i in range(sizeLogFilter):
        for j in range(sizeLogFilter):
            laplacianFilter[i][j] = log(i-a,j-b)  

    return convolution(img,laplacianFilter)

'''
Sharpening with high boost
img: Image that we use to apply high boost
'''
def highBoost(img):

    #Blur image
    tempImage = np.copy(img)
    tempImage= smoothing(tempImage)
    
    #Create mask
    mask = img-tempImage

    return np.uint8(highBoostParameter*mask)

'''
Sharpening with sobel operator
img: Image that we use to apply the sobel operator
'''
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

    #Convert
    Fx = np.float64(Fx)
    Fy = np.float64(Fy)
    
    return np.uint8(np.sqrt(np.power(Fx,2) + np.power(Fy,2)))
