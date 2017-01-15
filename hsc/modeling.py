
# Copyright (c) 2017, Simon Brodeur
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright 
#    notice, this list of conditions and the following disclaimer.
#   
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT 
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
# OF SUCH DAMAGE.

import logging
import itertools
import scipy
import scipy.sparse
import numpy as np
from numpy.lib.stride_tricks import as_strided

from hsc.utils import overlapAdd, overlapReplace, normalize

logger = logging.getLogger(__name__)

def extractRandomWindows(sequence, nbWindows, width):
    assert nbWindows > 0
    assert width > 0 and width < sequence.shape[0]
    
    # Generate random indices
    indices = np.random.randint(low=0, high=len(sequence)-width, size=(nbWindows,))
    
    # Create a strided view of the sequence and using indexing to extract windows
    s = as_strided(sequence, shape=(len(sequence)-width+1, width), strides=sequence.strides*2)
    windows = s[indices]
    return windows
    
def extractWindows(sequences, indices, width, centered=False):
    assert width > 0 and width < sequences.shape[1]
    
    # sequences has shape [batch, nbSamples, nbFeatures]

    # Create a strided view of the sequence and using indexing to extract windows
    s = as_strided(sequences, shape=(sequences.shape[0], sequences.shape[1]-width+1, width), strides=(sequences.strides[0], sequences.strides[1], sequences.strides[1]))

    i = np.arange(s.shape[0])
    if centered:
        if np.mod(width, 2) == 0:
            # Even
            windows = s[i, indices - (width/2-1)]
        else:
            # Odd
            windows = s[i, indices - (width/2)]
    else:
        windows = s[i, indices]
    return windows
    
# def kmeanPP(data, k, windowSize):
#     
#     # Start with an element chosen uniformly amongst data
#     means = extractRandomWindows(data, 1, windowSize)
#     for c in range(1, k):
# 
#         # Calculate the distance with existing centroids
#         distances = np.sum(tf.square(data[np.newaxis,:] - means[:, np.newaxis]), axis=2)
#         distances_nearest = np.max(distances, axis=0)
# 
#         # Select a new centroids randomly, proportional to the squared distance to the nearest centroid
#         prob = np.square(distances_nearest) / np.sum(np.square(distances_nearest))
# 
#         # Calculate the cumulative sum
#         # https://github.com/tensorflow/tensorflow/issues/813
#         # WARNING: this is not efficient!
#         cprob = tf.pad(prob, paddings=tf.expand_dims(tf.concat(0, [tf.shape(x)[0:1]-1,[0]]),0))
#         cprob = tf.reshape(cprob, [1,-1,1,1])
#         cfilter = tf.reshape(tf.ones_like(cprob), [-1,1,1,1])
#         cumsum = tf.nn.conv2d(cprob, cfilter, strides=[1,1,1,1], padding='VALID')
# 
#         # Multinomial sampling
#         idx = tf.cast(tf.reduce_min(tf.where(tf.greater_equal(cumsum, tf.random_uniform(tf.shape(prob))))), tf.int32)
#         mean = tf.slice(tf.random_shuffle(x), tf.concat(0, [tf.expand_dims(idx, 0), [0,]]), [1,-1,])
# 
#         # Add the centroid to codebook
#         means = tf.concat(0, [means, mean])
    
def convolve1d(sequence, filters, padding='valid'):

    sequence = np.atleast_2d(sequence).reshape((sequence.shape[0],-1))
    
    width = filters.shape[1]
    if padding == 'valid':
        # Nothing to do
        pass
    elif padding == 'same':
        # Pad sequences
        if np.mod(width, 2) == 0:
            # Even filter size
            sequence = np.pad(sequence, [(width/2-1, width/2), (0,0)], mode='reflect')
        else:
            # Odd filter size
            sequence = np.pad(sequence, [(width/2, width/2), (0,0)], mode='reflect')
    else:
        raise Exception('Padding not supported: %s' % (padding))

    if filters.ndim == 2:
        nbFeatures = 1
    else:
        nbFeatures = filters.shape[-1]
    assert nbFeatures == sequence.shape[-1]

    # Create a strided view of the sequence
    w = as_strided(sequence, ((sequence.shape[0] - width + 1), sequence.shape[1], width),
                  (sequence.strides[0], sequence.strides[1], sequence.strides[0])) 
    
    # w has shape [length, features, width]
    nbTotalFeatureDim = np.prod(filters.shape[1:])
    c = np.dot(w.reshape((w.shape[0],nbTotalFeatureDim)),
               filters.T.reshape(nbTotalFeatureDim, filters.shape[0]))
    return c

def convolve1d_batch(sequences, filters, padding='valid'):

    sequences = np.atleast_3d(sequences).reshape((sequences.shape[0], sequences.shape[1], -1))

    width = filters.shape[1]
    if padding == 'valid':
        # Nothing to do
        pass
    elif padding == 'same':
        # Pad sequences
        if np.mod(width, 2) == 0:
            # Even filter size
            sequences = np.pad(sequences, [(0,0), (width/2-1, width/2), (0,0)], mode='reflect')
        else:
            # Odd filter size
            sequences = np.pad(sequences, [(0,0), (width/2, width/2), (0,0)], mode='reflect')
    else:
        raise Exception('Padding not supported: %s' % (padding))
    
    if filters.ndim == 2:
        nbFeatures = 1
    else:
        nbFeatures = filters.shape[-1]
    assert nbFeatures == sequences.shape[-1]

    # Create a strided view of the sequence
    # byte_offset = stride[0]*index[0] + stride[1]*index[1] + ...
    # shape is (12, 20, 2), dtype is int64 , strides are (320, 16, 8)
    w = as_strided(sequences, (sequences.shape[0], (sequences.shape[1] - width + 1), sequences.shape[2], width),
                  (sequences.strides[0], sequences.strides[1], sequences.strides[2], sequences.strides[1]))
    
    # w has shape [batch, length, features, width, 1]
    nbTotalFeatureDim = np.prod(filters.shape[1:])
    c = np.dot(w.reshape((w.shape[0]*w.shape[1], nbTotalFeatureDim)),
               filters.T.reshape(nbTotalFeatureDim, filters.shape[0]))
    c = c.reshape(w.shape[0], w.shape[1], filters.shape[0])
    return c

class ConvolutionalDictionaryLearner(object):
 
    def __init__(self, k, windowSize, algorithm='kmean'):
        self.k = k
        self.windowSize = windowSize
        self.algorithm = algorithm
 
    def _train_samples(self, data):
        # Extract windows from the data of the same size as the dictionary
        windows = extractRandomWindows(data, self.k, self.windowSize)
 
        # Normalize the dictionary elements to have unit norms
        D = normalize(windows, axis=1)
        return D
 
    def _init_D(self, data, initMethod='random_samples'):
 
        if initMethod == 'noise':
            D = np.random.uniform(low=np.min(data), high=np.max(data), size=(self.k, self.windowSize))
        elif initMethod == 'random_samples':
            D = extractRandomWindows(data, self.k, self.windowSize)
        else:
            raise Exception('Unsupported initialization method: %s' % (initMethod))
        return D
 
    def _train_kmean(self, data, nbRandomWindows, maxIterations=100, tolerance=0.0, initMethod='random_samples', resetMethod='noise', nbAveragedPatches=8):
        # Reference:
        # Dundar, A., Jin, J., & Culurciello, E. (2016). Convolutional Clustering for Unsupervised Learning.
        # In ICLR. Retrieved from http://arxiv.org/abs/1511.06241

        # Extract windows from the data that are twice the length of the centroids
        windows = extractRandomWindows(data, nbRandomWindows, 2*self.windowSize)
 
        # Initialize dictionary with random windows from the data
        D = self._init_D(data, initMethod)

        n = 0
        alpha = tolerance + 1.0
        while n < maxIterations and alpha > tolerance:
 
            # Convolve the centroids with the windows
            innerProducts = convolve1d_batch(windows, D, padding='valid')
            # innerProducts has shape [batch, length, filters]
 
            # Find the maximum similarity amongst centroids inside each window
            indices = np.argmax(np.abs(innerProducts.reshape(innerProducts.shape[0], innerProducts.shape[1]*innerProducts.shape[2])), axis=1)
            indices = np.unravel_index(indices, (innerProducts.shape[1], innerProducts.shape[2]))
 
            # Extract the patch where there is maximum similarity, and assign it to the centroid.
            maxSampleIndices = indices[0]
            patches = extractWindows(windows, maxSampleIndices, width=D.shape[1], centered=True)
            assignments = indices[1]
            assert np.max(assignments) < D.shape[0]
 
            # Compute all centroids given the assigned patches
            def computeCentroid(c):
 
                # Detect empty prototypes and assign then to data examples if needed.
                # This is to avoid NaNs.
                assigned = np.where(assignments == c)
                if np.any(assigned):
                    # Assign to the mean of the patches
                    centroid = np.mean(patches[assigned], axis=0)
                else:
                    # Assign to the average of some random patches
                    if resetMethod == 'random_samples':
                        idx = np.random.randint(low=0, high=patches.shape[0])
                        centroid = patches[idx]
                    elif resetMethod == 'random_samples_average':
                        indices = np.random.randint(low=0, high=patches.shape[0], size=(nbAveragedPatches,))
                        centroid = np.mean(patches[indices], axis=0)
                    elif resetMethod == 'noise':
                        centroid = np.random.uniform(low=-1.0, high=1.0, size=patches.shape[1:])
                    else:
                        raise Exception('Unsupported reset method: %s' % (resetMethod))
 
                # Make sure that the centroid has not zero norm
                # TODO: maybe we could just add the epsilon in all cases?
                EPS = 1e-9
                l2norm = np.sqrt(np.sum(np.square(centroid)))
                if l2norm == 0.0:
                    centroid += EPS
 
                return centroid
 
            centroids = []
            for c in range(D.shape[0]):
                centroids.append(computeCentroid(c))
            centroids = np.vstack(centroids)
 
            # Normalize the centroids to have unit norms
            newD = normalize(centroids, axis=1)
 
            # Compute distance change
            alpha = np.sqrt(np.sum(np.square(D - newD)))
            
            D = newD
            n += 1
 
        return D
 
    def train(self, X, *args, **kwargs):
        if self.algorithm == 'samples':
            D = self._train_samples(X, *args, **kwargs)
        elif self.algorithm == 'kmean':
            D = self._train_kmean(X, *args, **kwargs)
        else:
            raise Exception('Unknown training algorithm: %s' % (self.algorithm))

        return D

class SparseApproximator(object):

    def computeCoefficients(self, X, D, dense=False):
        raise NotImplementedError()

class ConvolutionalMatchingPursuit(SparseApproximator):

    def __init__(self, nbNonzeroCoefs=None, tolerance=0.0):
        self.__dict__.update(nbNonzeroCoefs=nbNonzeroCoefs, tolerance=tolerance)

    def computeCoefficients(self, sequence, D):

        # Initialize the residual and sparse coefficients
        residual = np.copy(sequence)
        coefficients = scipy.sparse.lil_matrix((sequence.shape[0], D.shape[0]))
        
        # Convolve the input signal once, then locally recompute the affected projections when the residual is changed. 
        innerProducts = convolve1d(residual, D, padding='same')
        
        tolerance = self.tolerance + 1.0
        n = 0
        nbDuplicates = 0
        while tolerance > self.tolerance:

            # Find the maximum activation over the whole sequence and filters
            t, fIdx = np.unravel_index(np.argmax(np.abs(innerProducts)), innerProducts.shape)
            coefficient = innerProducts[t, fIdx]

            # Update the sparse coefficients
            if np.abs(coefficients[t, fIdx]) > 0.0:
                nbDuplicates += 1
            coefficients[t, fIdx] += coefficient

            # Update the residual by removing the contribution of the selected filter
            # NOTE: negate the coefficient to actually perform a overlap-remove operation.
            overlapAdd(residual, -coefficient*D[fIdx], t, copy=False)
            
            # Update the inner products
            # First, calculate the span of the residual that needs to be convolved again, 
            # and the padding required if at the beginning or end of the residual
            width = D.shape[1]
            padStart = 0
            if np.mod(width, 2) == 0:
                # Even width
                tstart = t-width/2+1-(width-1)
            else:
                # Odd width
                tstart = t-width/2-(width-1)
            startIdx = max(0,tstart)
            if tstart < 0:
                padStart = -tstart
                    
            tend = t+width/2+(width-1)
            endIdx = min(residual.shape[0]-1, tend)
            padEnd = 0
            if tend > residual.shape[0]-1:
                padEnd = tend - (residual.shape[0]-1)
            assert endIdx - startIdx >= 0
            assert padStart >= 0 and padEnd >= 0
             
            paddedResidual = np.pad(residual[startIdx:endIdx+1], [(padStart, padEnd)] + [(0,0) for _ in range(residual.ndim-1)], mode='reflect')
            localInnerProducts = convolve1d(paddedResidual, D, padding='valid')
            assert localInnerProducts.shape[0] == width + (width-1)
            overlapReplace(innerProducts, localInnerProducts, t, copy=False)
            
            tolerance = np.max(np.abs(residual))
            n += 1
            
            if self.nbNonzeroCoefs is not None and coefficients.nnz >= self.nbNonzeroCoefs:
                break

        logger.debug('Tolerance of %f achieved after %d iterations' % (tolerance, n))
        logger.debug('Number of duplicate coefficients: %d' % (nbDuplicates))

        return coefficients, residual

class ConvolutionalSparseCoder(object):
 
    def __init__(self, nbComponents, filterWidth, nbFeatures, approximator):
 
        # Initialize dictionary randomly, with all components having unit norms
        self.D = normalize(np.random.random(size=(nbComponents, filterWidth, nbFeatures)), axis=(1,2))
        
        self.approximator = approximator
 
    def encode(self, X):
        return self.approximator.computeCoefficients(X, self.D)
 
    def reconstruct(self, coefficients):
        assert scipy.sparse.issparse(coefficients)
        
        # Initialize signal
        signal = np.zeros((coefficients.shape[0], self.D.shape[-1]))
        
        # Iterate through all activations and overlap to signal
        cx = coefficients.tocoo()
        for t,fIdx,c in itertools.izip(cx.row, cx.col, cx.data):
            overlapAdd(signal, c*self.D[fIdx], t, copy=False)
        return signal
    
