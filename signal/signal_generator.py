# -*- coding: utf-8 -*-

"""
@file signal_generator.py
@brief signal generator
@author ふぇいと (@stfate)

@description

"""

def gen_white_noise(mean, var, length):
    if (method == 'uni'): # uniform noise
        noise = []
        for i in range(length):
            noise.append(random.uniform(-1, 1))
            
        return noise
        
    elif (method == 'gauss'): # gauss noise
        uni_noise1 = []
        uni_noise2 = []
        
        for i in range(length):
            uni_noise1.append(random.uniform(0, 1))
            uni_noise2.append(random.uniform(0, 1))
            
        gauss_noise1 = []
        gauss_noise2 = []
        for i in range(length):
            if (uni_noise1[i] == 0):
                y1 = 0
                y2 = 0
            else:
                y1 = sqrt(-2 * log(uni_noise1[i])) * cos(2*pi*uni_noise2[i])
                y2 = sqrt(-2 * log(uni_noise1[i])) * sin(2*pi*uni_noise2[i])
            gauss_noise1.append(y1)
            gauss_noise2.append(y2)
            
        abs_gn1 = []
        abs_gn2 = []
        for i in range(length):
            abs_gn1.append(abs(gauss_noise1[i]))
            abs_gn2.append(abs(gauss_noise2[i]))
            
        gn1_max = max(abs_gn1)
        gn2_max = max(abs_gn2)
        for i in range(length):
            gauss_noise1[i] = gauss_noise1[i] / gn1_max
            gauss_noise2[i] = gauss_noise2[i] / gn2_max
        
        return gauss_noise1