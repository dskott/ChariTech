# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:37:39 2017

@author: G.Gbatzolis, D. Scottis, H. Zhou

"""

from cross_validation import analysis
from image_processing import get_images
from Dimensional_Reduction_PCA import  dimension_reduction

extensions = ['A','L','R','T','W']
#extensions = ['R']
directory = './trainingSet/'

img_list_2, supervising_list = get_images(directory,extensions)

x_redu = dimension_reduction (img_list_2,2)

analysis(x_redu, supervising_list)

