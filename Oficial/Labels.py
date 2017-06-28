# -*- coding: utf-8 -*-
"""

@author: ramon
__authors__='Nuria Centellas(1395084), Hector Garcia(1391463), Miriam Traver(1391805)'
__group__='DJ15_05'

"""
import re
import numpy as np
import ColorNaming as cn
from skimage import color
import KMeans as km
from skimage import io
from heapq import nlargest

def NIUs():
    
    return 1395084, 1391463, 1391805

def loadGT(fileName):
    """@brief   Loads the file with groundtruth content
    
    @param  fileName  STRING    name of the file with groundtruth
    
    @return groundTruth LIST    list of tuples of ground truth data
                                (Name, [list-of-labels])
    """

    groundTruth = []
    fd = open(fileName, 'r')
    for line in fd:
        splitLine = line.split(' ')[:-1]
        labels = [''.join(sorted(filter(None,re.split('([A-Z][^A-Z]*)',l)))) for l in splitLine[1:]]
        groundTruth.append( (splitLine[0], labels) )
        
    return groundTruth


def evaluate(description, GT, options):
    """@brief   EVALUATION FUNCTION
    @param description LIST of color name lists: contain one lsit of color labels for every images tested
    @param GT LIST images to test and the real color names (see  loadGT)
    @options DICT  contains options to control metric, ...
    @return mean_score,scores mean_score FLOAT is the mean of the scores of each image
                              scores     LIST contain the similiraty between the ground truth list of color names and the obtained
    """
#########################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#########################################################
    # scores = np.random.rand(len(description),1)        
    # return sum(scores)/len(description), scores

    scores = []
    
    for i in range(len(description)):

        scores.append(similarityMetric(description[i], GT[i][1], options))

    return sum(scores)/len(description), scores


def similarityMetric(Est, GT, options):
    """@brief   SIMILARITY METRIC
    @param Est LIST  list of color names estimated from the image ['red','green',..]
    @param GT LIST list of color names from the ground truth
    @param options DICT  contains options to control metric, ...
    @return S float similarity between label LISTs
    """

#########################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#########################################################

    
    """if 'metric' in options:
        
        if options['metric'] == 'basic':
            intersect = 0
            for i in Est:
                if i in GT:
                    intersect += 1
            return float(intersect)/len(GT)
        
        elif options['metric'] == 'advanced':
            inte = 0
            for i in Est:
                if i in GT:
                    inte += 2
            return float(inte)/(len(GT)+len(Est))
        
        elif options['metric'] == 'misses':
            misses = 0
            for i in Est:
                if not i in GT:
                    misses += 1
            return float(misses)/len(Est)
        
    return 0"""
    
        
    """if options['metric'] == 'basic':
        intersect = np.intersect1d(Est, GT)
        
        similitudes = float(len(intersect))/float(len(GT))
        
        return similitudes

    elif options['metric'] == 'misses':

        similitudes = float(len(np.setdiff1d(GT, Est)))/float(len(GT))

        return similitudes

    elif options['metric'] == 'advanced':

        similitudes = float(len(np.setdiff1d(Est, GT)))/float(len(GT))

        return similitudes
    
    return 0"""
    if options['metric'].lower() == 'basic'.lower():
        x = 0
        for i in Est:
            for j in GT:
                if i == j:
                    x += 1
                    break
        S = x / float(len(Est))
        return S
    elif options['metric'] == 'advanced': # D'aqui pa vajo no afecta'l test (crec)
        x = 0
        for i in Est:
            if i in GT:
                x += 2
        return float(x)/(len(GT)+len(Est))

    elif options['metric'] == 'misses':
	misses = 0
	for i in Est:
	    if not i in GT:
		misses += 1
	return float(misses)/len(Est)
    else:    
    	return 0
   
def getLabels(kmeans, options):
    """@brief   Labels all centroids of kmeans object to their color names
    
    @param  kmeans  KMeans      object of the class KMeans
    @param  options DICTIONARY  options necessary for labeling
    
    @return colors  LIST    colors labels of centroids of kmeans object
    @return ind     LIST    indexes of centroids with the same color label
    """
    """
#########################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#########################################################

##  remind to create composed labels if the probability of 
##  the best color label is less than  options['single_thr']
    detected_colors = ['color'+'%d'%i for i in range(kmeans.K)]
    unique = range(kmeans.K)
    return detected_colors, unique
    """
  
    """detected_colors = []
    unique = []
    
    for i in range(len(kmeans.centroids)):
        
        cl = ''
        
        for j in range(len(kmeans.centroids[i])):
        
            if kmeans.centroids[i][j] > options['single_thr']:
            
                cl = cn.colors[j]

        if cl == '':
            
            auxlist = kmeans.centroids[i].tolist()
            color1 = cn.colors[auxlist.index(max(auxlist))]
            auxlist[auxlist.index(max(auxlist))] = 0
            color2 = cn.colors[auxlist.index(max(auxlist))]
            
            if color1 > color2:
                
                cl = color2 + color1
            
            else:
            
                cl = color1 + color2

        if cl in detected_colors:
            
            unique[detected_colors.index(cl)].append(i)
        
        elif cl != '':
        
            detected_colors.append(cl)
            unique.append([i])

    return detected_colors, unique"""
    
    detected_colors = []
    ordered_centroids = []  # unique=[[0,2][1]] si ens retornes 2 colors
   

    #Primer cal ordenar els centroides

    unique1, counts = np.unique(kmeans.clusters, return_counts=True) #Agafar cada valor i el numero de vegades que apareix
    order = counts.argsort()[::-1] #Ordenar la posisició (índex) dels valors de més aparicions a menos

    ordered_centroids = np.array([kmeans.centroids[i] for i in unique1[order]])

   #Un cop ordenats cal mirar si es volen colors independents o combos

    for i in ordered_centroids: 
       
        single = cn.colors[i.tolist().index(max(i))]

        if (max(i) >= options['single_thr']):
            detected_colors.append(single)
        else:
            # colorConjunts
            combos = []
            combos.append(cn.colors[i.tolist().index(nlargest(2, i)[0])])
            combos.append(cn.colors[i.tolist().index(nlargest(2, i)[1])])
            combos.sort()
            detected_colors.append(combos[0] + combos[1])

    ind = [] 
    colors = []
    count = 0
    for i in detected_colors:
        if i in colors:
            ind[colors.index(i)].append(count)
        else:
            ind.append([count])
            colors.append(i)
        count += 1
    return colors, ind

def processImage(im, options):#OJO
    """@brief   Finds the colors present on the input image
    
    @param  im      LIST    input image
    @param  options DICTIONARY  dictionary with options
    
    @return colors  LIST    colors of centroids of kmeans object
    @return indexes LIST    indexes of centroids with the same label
    @return kmeans  KMeans  object of the class KMeans
    """

#########################################################
##  YOU MUST ADAPT THE CODE IN THIS FUNCTIONS TO:
##  1- CHANGE THE IMAGE TO THE CORRESPONDING COLOR SPACE FOR KMEANS
##  2- APPLY KMEANS ACCORDING TO 'OPTIONS' PARAMETER
##  3- GET THE NAME LABELS DETECTED ON THE 11 DIMENSIONAL SPACE
#########################################################


##  1- CHANGE THE IMAGE TO THE CORRESPONDING COLOR SPACE FOR KMEANS
    if options['colorspace'].lower() == 'ColorNaming'.lower():
        im = np.reshape(im, (-1, im.shape[2]))
        im = cn.ImColorNamingTSELabDescriptor(im)
    elif options['colorspace'].lower() == 'RGB'.lower():
        im = np.reshape(im, (-1, im.shape[2]))
    elif options['colorspace'].lower() == 'Lab'.lower():        
        im = cn.RGB2Lab(im)

##  2- APPLY KMEANS ACCORDING TO 'OPTIONS' PARAMETER
    if options['K']<2: # find the best K
        kmeans = km.KMeans(im, 0, options)
        kmeans.bestK()
    else:
        kmeans = km.KMeans(im, options['K'], options) 
        kmeans.run()

##  3- GET THE NAME LABELS DETECTED ON THE 11 DIMENSIONAL SPACE
    if options['colorspace'].lower() == 'RGB'.lower():
        kmeans.centroids = cn.ImColorNamingTSELabDescriptor(kmeans.centroids)
#########################################################
##  THE FOLLOWING 2 END LINES SHOULD BE KEPT UNMODIFIED
#########################################################
    colors, which = getLabels(kmeans, options)   
    return colors, which, kmeans
