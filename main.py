#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 11:05:04 2016

@author: mirkodandrea
"""

import phast
import scipy.io as io

mat_file_rain = io.loadmat('test.mat')
R = mat_file_rain['a3dRadarRainT']    


phast.sigma = 0.05

output = phast.phast(R[:,:,-1], R[:,:,-2], n_ensamble=1)
