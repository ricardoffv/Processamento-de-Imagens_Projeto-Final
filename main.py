# -*- coding: utf-8 -*-

'''
Author: Lucas Yudi Sugi - 9293251
Discipline: SCC0251_1Sem_2018 
'''

import numpy as np
import imageio
import matplotlib.pyplot as plt
import filter
import resize
import segmentation

#Read the input image
def readImage():
    
    #Read the name of image
    nameImage = str(input()).rstrip();
    
    #Read the image
    img = imageio.imread(nameImage); 

    return img;

#Called for function that reads the image
img = readImage();

#temp image
tmpImage = np.copy(img)

#Pre-processing - Improving the image
tmpImage = filter.smoothing(tmpImage);
tmpImage = filter.histogramEqualizing(tmpImage);

#Pre-processing - Edge enhancement
#tmpImage = filter.laplacianOfGaussian(tmpImage);
img = filter.sobel(img);

#Resizes image
tmpImage = resize.resize(tmpImage)
img = resize.resize(img)

#Segmentation image
tmpImage = segmentation.segmentation(tmpImage);

#Apply the mask
img = np.multiply(tmpImage,img)

#Write image
imageio.imwrite('out.jpg',img)
