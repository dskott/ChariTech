# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 12:54:35 2017

@author: Demetris
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from skimage import color
import numpy

# Import image


def get_images(directory, extensions):
    'for the directory parameter use the local directory just before the folders A,L,R,T,W'
    img_list = []
    img_list_2 = []
    supervising_list = []
    j = 0
    for extension in extensions:
        contents = os.listdir(directory + extension)
        for image in contents:
            #transforms an image file into a vector 
            img_list.append(mpimg.imread(directory + extension+'/'+image))
            supervising_list.append(j)
        j+=1
            
    for image in img_list:
        img = color.rgb2gray(image)
        img_list_2.append(img.flatten())
        
    img_list_2 = numpy.array(img_list_2)
    
        
    return img_list_2, supervising_list

#answer = get_images(directory, extensions)
#print(answer[0][1][200])
#plt.imshow(answer[0])
#print(len(answer), type(answer[0]))  