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

#Read the input image
def readImage():
    
    #Read the name of image
    nameImage = str(input()).rstrip();
    
    #Read the image
    img = imageio.imread(nameImage); 

    return img;

#Called for function that reads the image
img = readImage();

#Pre-processing
#img = filter.smoothing(img);
#img = filter.gammaAdjustment(img);
#img = filter.histogramEqualizing(img);
#img = filter.highBoost(img);
img = filter.laplacianOfGaussian(img);
#img = filter.sobel(img);

#Resizes image
#img = resize.resize(img)

#Write image
imageio.imwrite('res.jpg',img)
#plt.imshow(img)
