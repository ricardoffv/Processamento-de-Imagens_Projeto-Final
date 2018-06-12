# -*- coding: utf-8 -*-

'''
Author: Lucas Yudi Sugi - 9293251
Discipline: SCC0251_1Sem_2018 
'''

import numpy as np
import imageio

#Dimensions of database image
row = 160
col = 160

#Resizes the image to 160x160 pixels
def resize(img):

    #dimensions of image
    M = img.shape[0]
    N = img.shape[1]
    
    #size of filter
    rowFilter = int(M/row)
    colFilter = int(N/col)

    #new dimensions
    newM = rowFilter * row
    newN = colFilter * col
    
    #cut image
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
    
    #Extract image
    tempImage = img[cutRow1:M-cutRow2,cutCol1:N-cutCol2,:]
    
    #the result image
    result = np.zeros([160,160,3])
    
    #reduce image
    for i in range(row):
        x = i*rowFilter
        for j in range(col):
            y = j*colFilter
            tempFilter = tempImage[x:x+rowFilter,y:y+colFilter]
            result[i][j][0] = np.mean(tempFilter[:,:,0])
            result[i][j][1] = np.mean(tempFilter[:,:,1])
            result[i][j][2] = np.mean(tempFilter[:,:,2])
    
    result = np.uint8(result)
    return result
