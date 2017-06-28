
# -*- coding: utf8 -*-
import KMeans as km
import numpy as np

# calcular la distancia entre 2 punts:

print("distancia")
print(km.distance(np.array([[57,280,95]]), np.array([[299,212,204]])))
print(km.distance(np.array([[57,280,95]]), np.array([[249,69,120]])))

# centroides:

print("2 - clusters")
X = np.array([[57,280,95], [131,237,79], [67,155,259], [85, 250, 74], [147, 250, 82], [96, 133, 88]])
C = np.array([[299,212,204], [249,69,120]])

a = km.KMeans(X, 2)
a.centroids = C
a._cluster_points()
print([i+1 for i in a.clusters])


# iteracio
print("3 - nous centroides")
X = np.array([[57,280,95], [131,237,79], [67,155,259], [85, 250, 74], [147, 250, 82], [96, 133, 88]])
C = np.array([[299,212,204], [249,69,120]])

a = km.KMeans(X, 2)
a.centroids = C
a._cluster_points()
a._get_centroids()
print(a.centroids)

# init
print("6 - inicialitzacio")

from skimage import io
ImageFolder = 'Images'
im = io.imread(ImageFolder+"/0061.jpg")
a = km.KMeans(im, 3)
a.run()
print(a.centroids)

# custom

print("7 - custom")
c = np.zeros((3, 1))
for k in range(3):
    c[k,:] = k*255/(3-1)
print(c)

# cielab
print("8 - cielab")

from skimage import io
ImageFolder = 'Images'
im = io.imread(ImageFolder+"/0061.jpg")
from skimage import color
im = color.rgb2lab(im)
print(im[62, 25])

# color naming
print("8 - cn")

from skimage import io
ImageFolder = 'Images'
im = io.imread(ImageFolder+"/0061.jpg")
import ColorNaming as cn
im = cn.ImColorNamingTSELabDescriptor(im)
print(im[83, 36])

