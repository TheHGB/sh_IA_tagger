
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

    return [[np.linalg.norm(x-c) for c in C] for x in X]

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
        self.K = K                                             # INT number of clusters

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
        distancia = distance(self.X, self.centroids)
        clusters = []
        n_clusters = 0

        for i in distancia:

            #clusters[n_clusters] = i.index(min(i)) # El cluster es el valor mes petit de la llista de distancies actual
            clusters.insert(n_clusters, i.index(min(i)))
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
        """
        newCentroids = {}
        
        for i in range(self.K):
        
            newCentroids[i] = []
        
        points = 0
        
        for i in self.clusters:
            
            newCentroids[i].append(self.X[points])
            
            points = points + 1
        
        self.old_centroids = np.copy(self.centroids)
        
        for i in range(self.K):
            
            self.centroids[i] = np.mean(newCentroids[i], axis = 0)"""
        newCentroids = []
        for i in range(self.K):
            aux = []
            aux = [self.X[j] for j in range(len(self.clusters)) if self.clusters[j] == i]
            if len(aux) != 0:
                aux2 = []
                for a in range(len(self.centroids[0])):
                    coordinates = 0
                    for b in aux:
                        coordinates += b[a]
                    dist = coordinates / len(aux)
                    aux2.append(dist)
                newCentroids.append(aux2)
            else:
                newCentroids.append(self.centroids[i])

        self.old_centroids = self.centroids
        self.centroids = np.array(newCentroids)
            

    def _converges(self):
        """@brief   Checks if there is a difference between current and old centroids
        """
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################

        """
        tolerancia = 0.6
        converged = True
        for i in range(self.K):
            dist = np.linalg.norm(self.centroids[i] - self.old_centroids[i])
            if dist > tolerancia:
                converged = False
                break
        return converged"""
    
        """tolerancia = self.options['tolerance']
        
        for i in range(self.K):
            #abso = distance(self.centroids[i], self.old_centroids[i])
            absoluto = abs(self.old_centroids[i] - self.centroids[i])
            np.allclose(absoluto, tolerancia)
        return True"""

        """absoluto = 0
        
        for i in range(len(self.centroids)):     

            for j in range(3):
                
                absoluto =+ np.sqrt((self.centroids[i][j]* self.options['tolerance'] - self.old_centroids[i][j])**2)
                
                return True
            
            return False"""

        for i in range(self.K):
            
            for j in range(len(self.centroids[i])):
                
                maxim = self.old_centroids[i][j] + (self.old_centroids[i][j] * self.options['tolerance'])
                minim = self.old_centroids[i][j] - (self.old_centroids[i][j] * self.options['tolerance'])
                
                if np.less(self.centroids[i][j], maxim).all() or np.greater(self.centroids[i][j], minim).all():
                
                    return False
                
        return True


        
    
        ##np.allclose(abs(self.old_centroids - self.centroids) <= ( self.options['tolerance']* abs(self.centroids)))
        
        #np.allclose(self.old_centroids, self.centroids, self.options['tolerance'])
        
        """coord = 0
    
        for i in range(self.K):
            
            absoluto = abs(self.old_centroids[i] - self.centroids[i]).all()
            
            coord =+ 1
            
            if (absoluto <= self.options['tolerance']):
                
                if coord == len(self.X) :
                    
                    return True
        return False"""
        
        #return (set([tuple(i) for i in self.centroids]) == set([tuple(i) for i in self.old_centroids]))
       
        
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
            self.bestK()
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
            
        K = 0
        
        if self.options['fitting'].lower() == 'fisher':
            
            self._init_rest(1)
            self.run()
            tmp_fitt = self.fitting()
            
            for i in range(float(self.options['max_iter'])):
                
                self._init_rest(i)
                self.run()
                tmp_fit = self.fitting()
                
                if tmp_fit < tmp_fitt:
                    
                    tmp_fitt = tmp_fit
                    K = i
                    
            return K
        
        else:
            
            for i in self.options['max_iter']:
                
                self._init_rest(i)
                self.run()
                
                if self.fitting()[0] >= 0.6:
                    #print("HOLAAAAAAAAAAAAAAAAAAAAAAAAAAA222222222222222") OK...
                    break
                    
            return K
            
            
    def fitting(self):
        """@brief  return a value describing how well the current kmeans fits the data
        """
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
        
        #COPIADO DEL FORUM

        if self.options['fitting'].lower() == 'fisher':
            betweenVariance = 0;
            medi = np.mean(self.X, 0)
            for i in range(0, self.K):
                matrix = self.X[self.clusters == i, :]
                medi2 = np.mean(self.matrix, 0)
                matrix = np.zeros((self.K, self.X.shape[-1]))
                matrix[i] = distance(medi2, medi)

            betweenVariance = matrix.mean(0)

        return betweenVariance
        


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
