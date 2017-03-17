# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 13:56:17 2017

@author: H.Zhou
"""

from Prin_Com_analysis import PCA_u

import numpy as np

def dimension_reduction(x,n):

    matrix_w = PCA_u(x,n)

    x_redu = x.dot(matrix_w)

    return(x_redu)