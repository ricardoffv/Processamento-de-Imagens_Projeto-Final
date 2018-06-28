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

#Dimensions of database's image
row = 160
col = 160

'''
Resizes the image to 160x160 pixels
img: Image that we want to resize
'''
def resize(img):

    #Dimensions of image
    M = img.shape[0]
    N = img.shape[1]
    
    #Size of filter (It's proportional to 160)
    rowFilter = int(M/row)
    colFilter = int(N/col)

    #New dimensions of cropped image
    newM = rowFilter * row
    newN = colFilter * col
    
    #Computes the limits of region that we want to cut
    if((M-newM)%2 == 0):
        cutRow1 = int((M-newM)/2)
        cutRow2 = cutRow1
    else:
        cutRow1 = int((M-newM)/2)
        cutRow2 = (M-newM)-cutRow1
    if((N-newN)%2 == 0):
        cutCol1 = int((N-newN)/2)
        cutCol2 = cutCol1
    else:
        cutCol1 = int((N-newN)/2)
        cutCol2 = (N-newN)-cutCol1
    
    #Extract image using the limits
    tempImage = img[cutRow1:M-cutRow2,cutCol1:N-cutCol2,:]
    
    #The result image resized
    result = np.zeros([160,160,3])
    
    #Reduce image
    for i in range(row):
        x = i*rowFilter
        for j in range(col):
            #Select one region for compute the mean color
            y = j*colFilter
            tempFilter = tempImage[x:x+rowFilter,y:y+colFilter]

            #Compute and set the RGB color
            result[i][j][0] = np.mean(tempFilter[:,:,0])
            result[i][j][1] = np.mean(tempFilter[:,:,1])
            result[i][j][2] = np.mean(tempFilter[:,:,2])
    
    #Convert for 1 byte
    result = np.uint8(result)

    return result
