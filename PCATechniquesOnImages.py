import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys


#Only deals with square PNG images and assumes they are greyscale    

class PCAImageProcessor():

    def __init__(self, images, dimension):
        self.images = images
        self.dimension = dimension
        self.numImages = len(self.images)
    
    def convertCompressedBackToSqrImg(self, compressed):
        indices = [x*250 for x in range(n)]
        return np.array([[[theList[x],theList[x],theList[x]] for x in range(i, i+self.dimension)] for i in indices])
    
    def arrangeVectorsByValues(self, values, vectors):        
        newOrder = sorted(range(len(values)), key=lambda i: values[i-1])
        return np.matrix([vectors[x] for x in newOrder])


class CompressedArray():
    def __init__(self, theArray, theFeatureVector, originalMeans):
        self.array = theArray
        self.featureVector = theFeatureVector
        self.originalMeans = originalMeans
        
    def decompress(self):
        normalizedResult = np.array(np.transpose(np.dot(self.featureVector.I, self.array)))    
        for k,mean in enumerate(self.originalMeans):
            normalizedResult[:,k] = self.originalMeans[k] + normalizedResult[:,k]
        return normalizedResult


class ImageBundleCompressor(PCAImageProcessor):
    
    def __init__(self, data, dimension):
        PCAImageProcessor.__init__(self, images, dimension)
        self.numImages = len(images)
        
    def convertToVectorRepOfImgList(self, imgList):
        vectors = list()        
        for x in range(self.dimension):
            for y in range(self.dimension):
                vectors += [[img[x][y][0] for img in self.images]]
        return np.array(vectors)
    
    def convertCompressedBackToSqrImg(self, theList, whichImageIndex=0):
        theList = theList[:,whichImageIndex]
        indices = [x*self.dimension for x in range(self.dimension)]
        return np.array([[[theList[x],theList[x],theList[x]] for x in range(i, i+250)] for i in indices])
        
    def compressImgArray(self, inclusionRate):
        includeReal = int(inclusionRate*self.numImages)
        
        theArray = self.convertToVectorRepOfImgList(self.images)
        
        meanValues = [np.mean(theArray[:,x]) for x in range(self.numImages)]
        adjustedData = np.array([column - mean for mean, column in zip(meanValues, theArray.T)])
        
        covarianceAdjData = np.cov(adjustedData)
        
        eigenValues, eigenVectors = np.linalg.eig(covarianceAdjData)
        featureVector = self.arrangeVectorsByValues(eigenValues, np.transpose(eigenVectors))
        
        result = np.dot(featureVector[:includeReal], adjustedData)
        return CompressedArray(result, featureVector[:includeReal], meanValues)


def showImages(original, new):
    fig = plt.figure(figsize=(12,10))
    for k,img in enumerate(original+new):
        a=fig.add_subplot(4,5,k)
        plt.imshow(img)
        plt.axis('off')
    plt.show()

images = [mpimg.imread(name) for name in ['pics/pic%s.png'%(s+1) for s in range(10)]]
inclusionRate =.9

IBC = ImageBundleCompressor(images, 250)
compressed = IBC.compressImgArray(inclusionRate)
result = compressed.decompress()
newImages = [IBC.convertCompressedBackToSqrImg(result, whichImageIndex=x) for x in range(10)]

showImages(images, newImages)
