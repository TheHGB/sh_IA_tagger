#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: ramon
"""
from skimage import io
import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import Labels as lb

def test_original():
    #'colorspace': 'RGB', 'Lab' o 'ColorNaming'
    options = {'colorspace':'RGB', 'K':6, 'synonyms':False, 'single_thr':0.6, 'verbose':False, 'km_init':'first', 'metric':'basic'}

    ImageFolder = 'Images'
    GTFile = 'LABELSsmall.txt'

    GTFile = ImageFolder + '/' + GTFile
    GT = lb.loadGT(GTFile)

    DBcolors = []
    for gt in GT:
        print gt[0]
        im = io.imread(ImageFolder+"/"+gt[0])
        colors,_,_ = lb.processImage(im, options)
        DBcolors.append(colors)

    encert,_ = lb.evaluate(DBcolors, GT, options)
    print "Encert promig: "+ '%.2f' % (encert*100) + '%'

def test_k(k, options):
    options['K'] = k

    ImageFolder = 'Images'
    #GTFile = 'LABELSlarge.txt'
    #GTFile = 'LABELSsmall.txt'
    GTFile = 'nye.txt'
    GTFile = ImageFolder + '/' + GTFile
    GT = lb.loadGT(GTFile)

    DBcolors = []
    for gt in GT:
        print gt[0]
        im = io.imread(ImageFolder+"/"+gt[0])
        colors,_,_ = lb.processImage(im, options)
        DBcolors.append(colors)

    return lb.evaluate(DBcolors, GT, options)

def test_ks():
    n0 = 2
    n = 12
    #res = []
    fig, ax = plt.subplots()
    ax.set_xticks(range(0, n+1))
    options = {'colorspace':'ColorNaming', 'synonyms':False, 'single_thr':0.6, 'verbose':False, 'km_init':'random', 'metric':'basic'}
    for k in range(n0, n+1):
        avg, scores = test_k(k, options)
       # plt.plot([k for _ in scores], scores, 'bo')
        plt.plot(k, avg, 'r^')
        #res.append(test_k(k))
    options = {'colorspace':'RGB', 'synonyms':False, 'single_thr':0.6, 'verbose':False, 'km_init':'random', 'metric':'basic'}
    for k in range(n0, n+1):
        avg, scores = test_k(k, options)
        #plt.plot([k+0.3 for _ in scores], scores, 'co')
        plt.plot(k, avg, 'g^')
        #res.append(test_k(k))
    options = {'colorspace':'Lab', 'synonyms':False, 'single_thr':0.6, 'verbose':False, 'km_init':'random', 'metric':'basic'}
    for k in range(n0,n+1): 
        avg, scores = test_k(k, options)
        #plt.plot([k+0.3 for _ in scores], scores, 'co')
        plt.plot(k, avg, 'b^')


    ax.set_ylabel("score")
    ax.set_xlabel("K")
    #ax.legend()
    plt.show()

if __name__ == "__main__":
    test_ks()
