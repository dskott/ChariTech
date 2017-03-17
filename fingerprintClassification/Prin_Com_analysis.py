# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 13:25:10 2017

Principle Component Analysis for the reduction of dimensions of the image data

@author: H. Zhou
"""

import numpy as np

#The function below applies principle component analysis on the matrix of training-
#set x of size m of diemnsion n to reduce to a dimension k

def PCA_u(x,k):

    n = len(x[0])

    u,s,v = np.linalg.svd(x.T)

#    return(u)
    
    u_comp = u[0:(k)]
    
#    return(u_comp)
    
    matrix_w = np.hstack((u_comp[0].reshape(n,1), u_comp[1].reshape(n,1)))
    
    return(matrix_w)
    
    
x = np.array([[1,2,3,3,4,6,5,4,7,8], [8,6,4,3,2,1,5,2,3,7], [3,4,5,2,8,4,4,4,5,5], [3,6,8,4,5,2,1,7,8,2]])   

n=2
    
#print(PCA_u(x,n))