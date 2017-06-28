
"""

@author: ramon
__authors__='Nuria Centellas(1395084), Hector Garcia(1391463), Miriam Traver(1391805)'
__group__='DJ15_05'

"""
import numpy as np
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from sklearn.decomposition import PCA


def distance(X,C):
    """@brief   Calculates the distance between each pixcel and each centroid 

    @param  X  numpy array PxD 1st set of data points (usually data points)
    @param  C  numpy array KxD 2nd set of data points (usually cluster centroids points)

    @return dist: PxK numpy array position ij is the distance between the 
    	i-th point of the first set an the j-th point of the second set
    """
#########################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#########################################################

    #return [[np.linalg.norm(x-c) for c in C] for x in X]
    dist = []
    for i in range(len(C)):
        dist.append(np.sqrt(np.sum(np.power(X-C[i],2),1)))
    return np.transpose(dist).tolist()


class KMeans():
    
    def __init__(self, X, K, options=None):
        """@brief   Constructor of KMeans class
        
        @param  X   LIST    input data
        @param  K   INT     number of centroids
        @param  options DICT dctionary with options
        """

        self._init_X(X)                                    # LIST data coordinates
        self._init_options(options)                        # DICT options
        self._init_rest(K)                                 # Initializes de rest of the object
        
#############################################################
##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
#############################################################

        
    def _init_X(self, X):
        """@brief Initialization of all pixels
        
        @param  X   LIST    list of all pixel values. Usually it will be a numpy 
                            array containing an image NxMx3

        sets X an as an array of data in vector form (PxD  where P=N*M and D=3 in the above example)
        """
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
        if X.ndim == 3:
            
            self.X = np.reshape(X, (-1, X.shape[2]))

        else:

            self.X = X
            

    def _init_options(self, options):
        """@brief Initialization of options in case some fields are left undefined
        
        @param  options DICT dctionary with options

			sets de options parameters
        """
        if options == None:
            options = {}
        if not 'km_init' in options:
            options['km_init'] = 'first'
        if not 'verbose' in options:
            options['verbose'] = False
        if not 'tolerance' in options:
            options['tolerance'] = 0
        if not 'max_iter' in options:
            options['max_iter'] = np.inf
        if not 'fitting' in options:
            options['fitting'] = 'Fisher'

        self.options = options
        
#############################################################
##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
#############################################################

        
    def _init_rest(self, K):
        """@brief   Initialization of the remainig data in the class.
        
        @param  options DICT dctionary with options
        """
        self.K = K
        if self.K>0:

            self._init_centroids()                             # LIST centroids coordinates

            self.old_centroids = np.empty_like(self.centroids) # LIST coordinates of centroids from previous iteration

            self.clusters = np.zeros(len(self.X))              # LIST list that assignes each element of X into a cluster

            self._cluster_points()                             # sets the first cluster assignation
	
        self.num_iter = 0                                      # INT current iteration
            
#############################################################
##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
#############################################################


    def _init_centroids(self):
        """@brief Initialization of centroids
        depends on self.options['km_init']
        """
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
        self.centroids = np.zeros(self.K)
        if self.options['km_init'] == 'first':
            self.centroids = self.X[:self.K] #Tots els primers pixels de la llista com centroides hi ha
        elif self.options['km_init'] == 'random':
            self.centroids = self.X[np.random.random_integers(0, len(self.X)-1, self.K)]
        else:
            self.centroids = self.X[np.random.random_integers(0, len(self.X)-1, self.K)]#RANDOM SENSE OPCIONS


        
    def _cluster_points(self):
        """@brief   Calculates the closest centroid of all points in X
        """
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
        distancia = distance(self.X, self.centroids) #Mirem la distancia de cada punt amb cada centroide
        clusters = []
        n_clusters = 0

        for i in distancia:
            clusters.insert(n_clusters, i.index(min(i)))#S'afegeix el punt al cluseter el centroide del qual esta mes proper
            n_clusters += 1
        self.clusters = np.array(clusters)


    def _get_centroids(self):
        """@brief   Calculates coordinates of centroids based on the coordinates 
                    of all the points assigned to the centroid
        """
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################

##       #Va mes rapid pero es perd precisio
#        self.old_centroids = np.copy(self.centroids)
#        for i in range(len(self.centroids)):
#            cluster_points = self.X[np.where(self.clusters == i)[0]]
#            if len(cluster_points) != 0:
#                self.centroids[i] = cluster_points.sum(axis=0)/len(cluster_points)
#

        self.old_centroids = self.centroids
        newCentroids = []
        for i in range(self.K):
            aux = [self.X[j] for j in range(len(self.clusters)) if self.clusters[j] == i] #Es fa una llista amb els clusters
            if len(aux) != 0:
                aux2 = []
                for a in range(len(self.centroids[0])): #Per a cada punt es mira la distancia al seu centroide i es fa la mitja
                    coordinates = 0
                    for b in aux:
                        coordinates += b[a]
                    dist = coordinates / len(aux)
                    aux2.append(dist)
                newCentroids.append(aux2)
            else:
                newCentroids.append(self.centroids[i]) 

        self.centroids = np.array(newCentroids)#No hi ha hagut nassos d'anar afegint a un array de numpy, s'ha fet amb una llista





    def _converges(self):
        """@brief   Checks if there is a difference between current and old centroids
        """
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################

        for centroids, old in zip(self.centroids, self.old_centroids): #Recorrem simultanieament els nous i els vells centroides i comparem amb la tolerancia
            for i in range(len(centroids)):
                if abs(centroids[i] - old[i]) > self.options['tolerance']:
                    return False
        return True
#	for i in range(self.K):
#            
#            for j in range(len(self.centroids[i])):
#                
#                maxim = self.old_centroids[i][j] + (self.old_centroids[i][j] * self.options['tolerance'])
#                minim = self.old_centroids[i][j] - (self.old_centroids[i][j] * self.options['tolerance'])
#                
#                if np.less(self.centroids[i][j], maxim).all() or np.greater(self.centroids[i][j], minim).all():
#                
#                    return False
#                
#        return True
       
        
    def _iterate(self, show_first_time=True):
        """@brief   One iteration of K-Means algorithm. This method should 
                    reassigne all the points from X to their closest centroids
                    and based on that, calculate the new position of centroids.
        """
        self.num_iter += 1
        self._cluster_points()
        self._get_centroids()
        if self.options['verbose']:
            self.plot(show_first_time)


    def run(self):
        """@brief   Runs K-Means algorithm until it converges or until the number
                    of iterations is smaller than the maximum number of iterations.=
        """
        if self.K==0:
            self.K == self.bestK()
            return        
        
        self._iterate(True)
        self.options['max_iter'] = np.inf
        if self.options['max_iter'] > self.num_iter:
            while not self._converges() :
                self._iterate(False)
      
      
    def bestK(self):
        """@brief   Runs K-Means multiple times to find the best K for the current 
                    data given the 'fitting' method. In cas of Fisher elbow method 
                    is recommended.
                    
                    at the end, self.centroids and self.clusters contains the 
                    information for the best K. NO need to rerun KMeans.
           @return B is the best K found.
        """
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
        fitting = np.zeros(11)#Agafem un maxim possible de 11 centroides (coincideix amb l'espai de color major que utilitzem)
        for i in range(2,12):
            self._init_rest(i)
            self.run()
            fitting[i-2] = self.fitting()
            if i > 1:
                if 0.2*abs(fitting[(i-2)-2]-fitting[(i-2)-1]) > abs(fitting[(i-2)-1]-fitting[i-2]):#Controlar la variacio amb un 20% per cent tolerancia
                        return i
        return np.argmin(fitting)#Agafem el valor mrimim si s'ha evitat la tolerancia amb tot



    def fitting(self):
        """@brief  return a value describing how well the current kmeans fits the data
        """
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
        #CODI BASAT EN IMPLEMENTACIONS DE FISHER TROBADES PER INTERNET I LA TEORIA FETA A CLASSE
        def lafunsionsita(a):#Amb el codi final no fa falta, pero em va costar la vida pensar-ho i no em dona la gana desfer-me'n
                return 1.0/a
        lafunsionsita = np.vectorize(lafunsionsita)
        if self.options['fitting'].lower() == 'fisher':
            interclass = 0.0
            for i, centroide in enumerate(self.centroids):#Guardar el centroide i la posicio
                for pixel in self.X[self.clusters == i]:
                    interclass +=  np.linalg.norm(centroide-pixel)
            meh, repeticions = np.unique(self.clusters, return_counts = True) #Guardar el nobre d'aparicions de cada centroide. meh necessari because of reasons
            suma = sum(lafunsionsita(repeticions).tolist())
            interclass = interclass * suma * (1.0/self.K)
            intraclass = 0.0
            centroidazo = np.average(self.centroids)
            for centroide in self.centroids: #Mirar si dona temps a ver si podem recilcar la llista de centroides per estalviar temps
                intraclass +=  np.linalg.norm(centroide - centroidazo)
            intraclass = intraclass*(1.0/self.K)
            return intraclass/interclass
        else:
            return np.random.rand(1)#Si no volen fisher un random i a tope de power


    def plot(self, first_time=True):
        """@brief   Plots the results
        """

        #markersshape = 'ov^<>1234sp*hH+xDd'	
        markerscolor = 'bgrcmybgrcmybgrcmyk'
        if first_time:
            plt.gcf().add_subplot(111, projection='3d')
            plt.ion()
            plt.show()

        if self.X.shape[1]>3:
            if not hasattr(self, 'pca'):
                self.pca = PCA(n_components=3)
                self.pca.fit(self.X)
            Xt = self.pca.transform(self.X)
            Ct = self.pca.transform(self.centroids)
        else:
            Xt=self.X
            Ct=self.centroids

        for k in range(self.K):
            plt.gca().plot(Xt[self.clusters==k,0], Xt[self.clusters==k,1], Xt[self.clusters==k,2], '.'+markerscolor[k])
            plt.gca().plot(Ct[k,0:1], Ct[k,1:2], Ct[k,2:3], 'o'+'k',markersize=12)

        if first_time:
            plt.xlabel('dim 1')
            plt.ylabel('dim 2')
            plt.gca().set_zlabel('dim 3')
        plt.draw()
        plt.pause(0.01)
