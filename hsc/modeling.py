
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

import os
import logging
import tempfile
import itertools
import copy
import collections
import scipy
import scipy.signal
import scipy.sparse
import numpy as np
from numpy.lib.stride_tricks import as_strided

import matplotlib
import matplotlib.pyplot as plt

from hsc.utils import overlapAdd, overlapReplace, normalize, peek, findGridSize
from hsc.dataset import MultilevelDictionary

logger = logging.getLogger(__name__)

def extractRandomWindows(sequence, nbWindows, width):
    assert sequence.ndim == 1 or sequence.ndim == 2
    assert nbWindows > 0
    assert width > 0 and width < sequence.shape[0]
    
    # Generate random indices
    indices = np.random.randint(low=0, high=sequence.shape[0]-width, size=(nbWindows,))
    return extractWindows(sequence, indices, width, centered=False)
    
def extractWindows(sequence, indices, width, centered=False):
    assert sequence.ndim == 1 or sequence.ndim == 2
    assert width > 0 and width < sequence.shape[0]

    if sequence.ndim == 1:
        sequence = sequence[:,np.newaxis]
        squeezeOutput = True
    else:
        squeezeOutput = False
    
    # Create a strided view of the sequence and using indexing to extract windows
    s = as_strided(sequence, shape=((sequence.shape[0] - width + 1), width, sequence.shape[1]),
                   strides=(sequence.strides[0], sequence.strides[0], sequence.strides[1]))
    
    if centered:
        if np.mod(width, 2) == 0:
            # Even
            windows = s[indices - (width/2-1)]
        else:
            # Odd
            windows = s[indices - (width/2)]
    else:
        windows = s[indices]
        
    if squeezeOutput:
        windows = np.squeeze(windows, axis=2)
    return windows
    
def extractWindowsBatch(sequences, indices, width, centered=False):
    assert sequences.ndim == 2 or sequences.ndim == 3
    assert width > 0 and width <= sequences.shape[1]
    
    if sequences.ndim == 2:
        sequences = sequences[:,:,np.newaxis]
        squeezeOutput = True
    else:
        squeezeOutput = False
    
    # Create a strided view of the sequence and using indexing to extract windows
    s = as_strided(sequences, shape=(sequences.shape[0], (sequences.shape[1] - width + 1), width, sequences.shape[2]),
                  strides=(sequences.strides[0], sequences.strides[1], sequences.strides[1], sequences.strides[2]))

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
        
    if squeezeOutput:
        windows = np.squeeze(windows, axis=2)
        
    return windows
    
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
            sequence = np.pad(sequence, [(width/2-1, width/2), (0,0)], mode='constant')
        else:
            # Odd filter size
            sequence = np.pad(sequence, [(width/2, width/2), (0,0)], mode='constant')
    else:
        raise Exception('Padding not supported: %s' % (padding))

    if filters.ndim == 2:
        nbFeatures = 1
    else:
        nbFeatures = filters.shape[-1]
    assert nbFeatures == sequence.shape[-1]

    # TODO: when to use scipy.signal.fftconvolve?
    # convolve fiters by the signal
    # TODO: broadcast sequence over filter axis to avoid the loop
#     c = [scipy.signal.fftconvolve(sequence, filter[::-1, ::-1], mode='valid') for filter in filters]
#     c = np.squeeze(np.stack(c, axis=-1), axis=1)
    
    # Create a strided view of the sequence
    w = as_strided(sequence, shape=((sequence.shape[0] - width + 1), sequence.shape[1], width),
                  strides=(sequence.strides[0], sequence.strides[1], sequence.strides[0])) 
     
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
            sequences = np.pad(sequences, [(0,0), (width/2-1, width/2), (0,0)], mode='constant')
        else:
            # Odd filter size
            sequences = np.pad(sequences, [(0,0), (width/2, width/2), (0,0)], mode='constant')
    else:
        raise Exception('Padding not supported: %s' % (padding))
    
    if filters.ndim == 2:
        nbFeatures = 1
    else:
        nbFeatures = filters.shape[-1]
    assert nbFeatures == sequences.shape[-1]

    # Create a strided view of the sequence
    w = as_strided(sequences, shape=(sequences.shape[0], (sequences.shape[1] - width + 1), sequences.shape[2], width),
                  strides=(sequences.strides[0], sequences.strides[1], sequences.strides[2], sequences.strides[1]))
    
    # w has shape [batch, length, features, width, 1]
    nbTotalFeatureDim = np.prod(filters.shape[1:])
    c = np.dot(w.reshape((w.shape[0]*w.shape[1], nbTotalFeatureDim)),
               filters.T.reshape(nbTotalFeatureDim, filters.shape[0]))
    c = c.reshape(w.shape[0], w.shape[1], filters.shape[0])
    return c

def reconstructSignal(coefficients, D):
    assert coefficients.ndim == 1 or coefficients.ndim == 2
    assert D.ndim == 2 or D.ndim == 3
    
    # D should be a 3D tensor, with the last dimension being the number of features
    if D.ndim == 2:
        squeezeOutput = True
        D = D[:,:,np.newaxis]
    else:
        squeezeOutput = False
    
    # Initialize signal
    signal = np.zeros((coefficients.shape[0], D.shape[-1]), dtype=coefficients.dtype)
    
    if scipy.sparse.issparse(coefficients):
        # Iterate through all sparse activations and overlap to signal
        cx = coefficients.tocoo()
        for t,fIdx,c in itertools.izip(cx.row, cx.col, cx.data):
            overlapAdd(signal, c*D[fIdx], t, copy=False)
    else:
        # Dense convolution (in Fourier domain)
        filterWidth = D.shape[1]
        if np.mod(filterWidth, 2) == 0:
            # Even
            offset = filterWidth/2-1
        else:
            # Odd
            offset = filterWidth/2

        # Iterate through all bases of the dictionary
        for basis, activation in zip(D, coefficients.T):
            signal += scipy.signal.fftconvolve(basis.T, np.atleast_2d(activation), mode='full').T[offset:offset + coefficients.shape[0]]
    
    if squeezeOutput:
        signal = np.squeeze(signal, axis=1)
        
    return signal

class ConvolutionalDictionaryLearner(object):
 
    def __init__(self, k, windowSize, algorithm='kmean', avoidSingletons=False, verbose=False):
        self.k = k
        self.windowSize = windowSize
        self.algorithm = algorithm
        self.avoidSingletons = avoidSingletons
 
        if verbose:
            plt.ion()
            self.fig = plt.figure(figsize=(8,8), facecolor='white', frameon=True)
        else:
            self.fig = None
        self.verbose = verbose
 
    def _train_samples(self, data):
        
        # Loop until all the required number of non-null patterns is found
        patterns = []
        nbPatternsFound = 0
        while nbPatternsFound < self.k:
        
            # Extract windows from the data of the same size as the dictionary
            windows = extractRandomWindows(data, self.k, self.windowSize)
     
            # Check if windows contain patterns that have non-zero norms
            l2norms = np.sqrt(np.sum(np.square(windows), axis=tuple(range(1, windows.ndim))))
            if self.avoidSingletons:
                l0norms = np.sum(windows != 0.0, axis=tuple(range(1, windows.ndim)))
                validWindows = windows[np.where((l2norms > 0.0) & (l0norms > 1))]
            else:
                validWindows = windows[np.where(l2norms > 0.0)]
            
            for validWindow in validWindows:
                patterns.append(validWindow)
                nbPatternsFound += 1
                if nbPatternsFound == self.k:
                    break
                
        patterns = np.stack(patterns)
                
        # Normalize the dictionary elements to have unit norms
        D = normalize(patterns)
            
        return D
 
    def _init_D(self, data, initMethod='random_samples'):
        assert data.ndim == 1 or data.ndim == 2
        
        if data.ndim == 1:
            squeezeOutput = True
            data = data[:,np.newaxis]
        else:
            squeezeOutput = False
 
        if initMethod == 'noise':
            D = normalize(np.random.uniform(low=np.min(data), high=np.max(data), size=(self.k, self.windowSize, data.shape[-1])))
        elif initMethod == 'random_samples':
            D = normalize(extractRandomWindows(data, self.k, self.windowSize))
        else:
            raise Exception('Unsupported initialization method: %s' % (initMethod))
        
        if squeezeOutput:
            D = np.squeeze(D, axis=2)
        
        return D
 
    def _train_nmf(self, data, initMethod='random_samples', nbMaxIterations=None, toleranceResidualScale=None, toleranceSnr=None):
        
        # Initialize dictionary from the data
        D = self._init_D(data, initMethod)
        sequence = data

        if sequence.ndim == 1:
            sequence = sequence[:,np.newaxis]
        if D.ndim == 2:
            D = D[:,:,np.newaxis]

        energySignal = np.sum(np.square(sequence))

        # Initialize the coefficients
        coefficients = np.random.random((sequence.shape[0], D.shape[0])).astype(sequence.dtype) + 2.0

        filterWidth = D.shape[1]
        if np.mod(filterWidth, 2) == 0:
            # Even
            offsetStart = filterWidth/2-1
            offsetEnd = filterWidth/2
            cutEndOffset = filterWidth - 1
        else:
            # Odd
            offsetStart = filterWidth/2
            offsetEnd = filterWidth/2
            cutEndOffset = filterWidth - 1
            
        # Loop until convergence or if any stopping criteria is met
        nbIterations = 0
        converged = False
        while not converged:
        
            # Update coefficients
            # Adapted from: https://github.com/mattwescott/senmf
            for t in range(D.shape[1]):
                # Calculate multiplicative residual
                coefficientsCentered = np.pad(coefficients[:-filterWidth+1], [(offsetStart,offsetEnd)] + [(0,0) for _ in range(coefficients.ndim-1)], mode='constant')
                reconstruction = reconstructSignal(coefficientsCentered, D)
                R = sequence / np.abs(reconstruction)
                R = np.pad(R[t:], [(0,t)] + [(0,0) for _ in range(R.ndim-1)], mode='constant')
                U_A = (np.dot(D[:,t,:], R.T) / np.sum(D[:,t,:], axis=1, keepdims=True)).T
                coefficients *= U_A
            
            # Update dictionary
            # Adapted from: https://github.com/mattwescott/senmf
            coefficientsCentered = np.pad(coefficients[:-filterWidth+1], [(offsetStart,offsetEnd)] + [(0,0) for _ in range(coefficients.ndim-1)], mode='constant')
            reconstruction = reconstructSignal(coefficientsCentered, D)
            R = sequence / np.abs(reconstruction)
            D_updates = np.zeros_like(D)
            for t in range(D.shape[1]):
                U_D = np.dot(coefficients[:-t or None,:].T, R[t:]) / coefficients[:-t or None,:].sum(axis=0, keepdims=True).T
                D_updates[:,t,:] = U_D
            D *= D_updates
            
            # Normalize dictionary
            D = normalize(D)
            
            # Calculate additive residual
            coefficientsCentered = np.pad(coefficients[:-filterWidth+1], [(offsetStart,offsetEnd)] + [(0,0) for _ in range(coefficients.ndim-1)], mode='constant')
            reconstruction = reconstructSignal(coefficientsCentered, D)
            residual = sequence - reconstruction
            
            # Print information about current iteration
            residualScale = np.max(np.abs(residual))
            energyResidual = np.sum(np.square(residual))
            snr = 10.0*np.log10(energySignal/energyResidual)
            logger.debug('SNR of %f dB achieved after %d iterations' % (snr, nbIterations))
            nbIterations += 1
            
            # Check stopping criteria
            if nbIterations is not None and nbIterations >= nbMaxIterations:
                logger.debug('Maximum number of iterations reached')
                converged = True
                break
            if toleranceResidualScale is not None and residualScale <= toleranceResidualScale:
                logger.debug('Tolerance for residual scale (absolute value) reached')
                converged = True
                break
            if toleranceSnr is not None and snr >= toleranceSnr:
                logger.debug('Tolerance for signal-to-noise ratio reached')
                converged = True
                break
 
        if data.ndim == 1:
            D = np.squeeze(D, axis=2)
 
        return D
 
    def _train_kmean(self, data, nbRandomWindows, maxIterations=100, tolerance=0.0, initMethod='random_samples', resetMethod='noise', nbAveragedPatches=8):
        # Reference:
        # Dundar, A., Jin, J., & Culurciello, E. (2016). Convolutional Clustering for Unsupervised Learning.
        # In ICLR. Retrieved from http://arxiv.org/abs/1511.06241

        # Extract windows from the data that are twice the length of the centroids
        windows = extractRandomWindows(data, nbRandomWindows, 2 * self.windowSize)
 
        # Initialize dictionary with random windows from the data
        D = self._init_D(data, initMethod)

        n = 0
        alpha = tolerance + 1.0
        while n < maxIterations and alpha > tolerance:
 
            if self.verbose:
                count = D.shape[0]
                m,k = findGridSize(count)
                idx = 0
                self.fig.clf()
                self.fig.canvas.set_window_title('Iteration no.%d' % (n))
                for i in range(m):
                    for j in range(k):
                        ax = self.fig.add_subplot(m,k,idx+1)
                        ax.plot(D[idx], linewidth=2, color='k')
                        r = np.max(np.abs(D[idx]))
                        ax.set_ylim(-r, r)
                        ax.set_axis_off()
                        idx += 1
                        if idx >= count:
                            break
                
                self.fig.canvas.draw()
 
            # Convolve the centroids with the windows
            innerProducts = convolve1d_batch(windows, D, padding='valid')
            # innerProducts has shape [batch, length, filters]
 
            # Find the maximum similarity amongst centroids inside each window
            indices = np.argmax(np.abs(innerProducts.reshape(innerProducts.shape[0], -1)), axis=1)
            indices = np.unravel_index(indices, (innerProducts.shape[1], innerProducts.shape[2]))
            
            # Extract the patch where there is maximum similarity, and assign it to the centroid.
            maxSampleIndices = indices[0]
            filterWidth = D.shape[1]
            if np.mod(filterWidth, 2) == 0:
                # Even
                offset = filterWidth/2 - 1
            else:
                # Odd
                offset = filterWidth/2
            patches = extractWindowsBatch(windows, offset + maxSampleIndices, width=D.shape[1], centered=True)
            assignments = indices[1]
            assert np.max(assignments) < D.shape[0]
 
            # Compute all centroids given the assigned patches
            def computeCentroid(c):
 
                # Detect empty prototypes and assign then to data examples if needed.
                # This is to avoid NaNs.
                assigned = np.where(assignments == c)
                if np.any(assigned):
                    # Assign to the mean of the normalized patches (i.e. cosine mean)
                    centroid = np.mean(normalize(patches[assigned]), axis=0)
                    isReset = False
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
                    isReset = True
                    
                # Make sure that the centroid has not zero norm
                EPS = 1e-9
                l2norm = np.sqrt(np.sum(np.square(centroid)))
                if l2norm == 0.0:
                    centroid += EPS
 
                return centroid, isReset
 
            nbResets = 0
            centroids = []
            for c in range(D.shape[0]):
                centroid, isReset = computeCentroid(c)
                if isReset:
                    nbResets += 1
                centroids.append(centroid)
            centroids = np.stack(centroids)
 
            # Normalize the centroids to have unit norms
            newD = normalize(centroids)
 
            # Compute distance change
            alpha = np.sqrt(np.sum(np.square(D - newD)))
            
            logger.debug('K-mean iteration %d: tolerance = %f, nb resets = %d' % (n, alpha, nbResets))
            D = newD
            n += 1
 
        return D
 
    def train(self, X, *args, **kwargs):
        if self.algorithm == 'samples':
            D = self._train_samples(X, *args, **kwargs)
        elif self.algorithm == 'kmean':
            D = self._train_kmean(X, *args, **kwargs)
        elif self.algorithm == 'nmf':
            D = self._train_nmf(X, *args, **kwargs)
        else:
            raise Exception('Unknown training algorithm: %s' % (self.algorithm))

        return D

class SparseApproximator(object):

    def computeCoefficients(self, X, D):
        raise NotImplementedError()

class ConvolutionalNMF(SparseApproximator):

    def __init__(self):
        pass

    def computeCoefficients(self, sequence, D, nbMaxIterations=None, toleranceResidualScale=None, toleranceSnr=None):
        assert sequence.ndim == 1 or sequence.ndim == 2
        assert D.ndim == 2 or D.ndim == 3

        if sequence.ndim == 1:
            squeezeOutput = True
            sequence = sequence[:,np.newaxis]
        else:
            squeezeOutput = False
            
        if D.ndim == 2:
            squeezeOutput = True
            D = D[:,:,np.newaxis]

        energySignal = np.sum(np.square(sequence))

        # Initialize the coefficients
        coefficients = np.random.random((sequence.shape[0], D.shape[0])).astype(sequence.dtype) + 2.0

        filterWidth = D.shape[1]
        if np.mod(filterWidth, 2) == 0:
            # Even
            offsetStart = filterWidth/2-1
            offsetEnd = filterWidth/2
            cutEndOffset = filterWidth - 1
        else:
            # Odd
            offsetStart = filterWidth/2
            offsetEnd = filterWidth/2
            cutEndOffset = filterWidth - 1
            
        # Loop until convergence or if any stopping criteria is met
        nbIterations = 0
        converged = False
        while not converged:
        
            # Update coefficients
            # Adapted from: https://github.com/mattwescott/senmf
            for t in range(D.shape[1]):
                
                # Calculate multiplicative residual
                coefficientsCentered = np.pad(coefficients[:-filterWidth+1], [(offsetStart,offsetEnd)] + [(0,0) for _ in range(coefficients.ndim-1)], mode='constant')
                reconstruction = reconstructSignal(coefficientsCentered, D)
                R = sequence / np.abs(reconstruction)
                R = np.pad(R[t:], [(0,t)] + [(0,0) for _ in range(R.ndim-1)], mode='constant')
                
                U_A = (np.dot(D[:,t,:], R.T) / np.sum(D[:,t,:], axis=1, keepdims=True)).T
                coefficients *= U_A
            
            # Calculate additive residual
            coefficientsCentered = np.pad(coefficients[:-filterWidth+1], [(offsetStart,offsetEnd)] + [(0,0) for _ in range(coefficients.ndim-1)], mode='constant')
            reconstruction = reconstructSignal(coefficientsCentered, D)
            residual = sequence - reconstruction
            
            # Print information about current iteration
            residualScale = np.max(np.abs(residual))
            energyResidual = np.sum(np.square(residual))
            snr = 10.0*np.log10(energySignal/energyResidual)
            logger.debug('SNR of %f dB achieved after %d iterations' % (snr, nbIterations))
            nbIterations += 1
            
            # Check stopping criteria
            if nbIterations is not None and nbIterations >= nbMaxIterations:
                logger.debug('Maximum number of iterations reached')
                converged = True
                break
            if toleranceResidualScale is not None and residualScale <= toleranceResidualScale:
                logger.debug('Tolerance for residual scale (absolute value) reached')
                converged = True
                break
            if toleranceSnr is not None and snr >= toleranceSnr:
                logger.debug('Tolerance for signal-to-noise ratio reached')
                converged = True
                break
    
        coefficients = coefficientsCentered
    
        if squeezeOutput:
            residual = np.squeeze(residual, axis=1)
    
        return coefficients, residual

class MptkConvolutionalMatchingPursuit(SparseApproximator):

    def __init__(self, method='mp'):
        self.method = method

    def _createMptkDict(self, D):

        # Save MPTK table data to binary file (Row-major layout, double precision)
        fd, dataFilePath = tempfile.mkstemp(suffix='.bin')
        os.close(fd)
        np.array(D, order='C', dtype=np.float64).tofile(dataFilePath)

        # Save MPTK table definition to XML file
        tableXml = """<?xml version="1.0" encoding="ISO-8859-1"?>
        <table>
        <libVersion>0.7.0</libVersion>
        <param name="numChans" value="%i"/>
        <param name="filterLen" value="%i"/>
        <param name="numFilters" value="%i"/>
        <param name="data" value="%s"/>
        </table>
        """ % (D.shape[2], D.shape[1], D.shape[0], dataFilePath)
        fd, tableFilePath = tempfile.mkstemp(suffix='.xml')
        os.write(fd, tableXml)
        os.close(fd)
        
        # Save MPTK dictionary definition to XML file
        dictXml = """<?xml version="1.0" encoding="iso-8859-1" ?>
        <dict>
        <libVersion>0.7.0</libVersion>
        <block>
        <param name="type" value="anywave" />
        <param name="tableFileName" value="%s" />
        <param name="windowShift" value="1" />
        </block>
        </dict>
        """ % (tableFilePath)
        fd, dictFilePath = tempfile.mkstemp(suffix='.xml')
        os.write(fd, dictXml)
        os.close(fd)
        
        return dictFilePath, tableFilePath, dataFilePath

    def computeCoefficients(self, sequence, D, nbNonzeroCoefs=None, toleranceSnr=None):
        assert sequence.ndim == 1 or sequence.ndim == 2
        assert D.ndim == 2 or D.ndim == 3

        import mptk
        mptk.loadconfig('/usr/local/mptk/path.xml')

        if sequence.ndim == 1:
            squeezeOutput = True
            sequence = sequence[:,np.newaxis]
        else:
            squeezeOutput = False
            
        if D.ndim == 2:
            squeezeOutput = True
            D = D[:,:,np.newaxis]

        try:
            fs = np.min([D.shape[1], sequence.shape[0]])
            dictFilePath, tableFilePath, dataFilePath = self._createMptkDict(D)
            if toleranceSnr is not None:
                book, residual = mptk.decompose(sequence, dictFilePath, fs, snr=toleranceSnr, method=self.method)
            elif nbNonzeroCoefs is not None:
                book, residual = mptk.decompose(sequence, dictFilePath, fs, numiters=nbNonzeroCoefs, method=self.method)
            
            # Get the sparse activations from the MPTK book
            fIndices = np.array([atom['anywaveIdx'] for atom in book.atoms], dtype=np.int)
            positions = np.array([atom['pos'][0] for atom in book.atoms], dtype=np.int)
            amplitudes = np.array([atom['amp'][0] for atom in book.atoms], dtype=np.float32)
            
            # Create sparse coefficient matrix
            coefficients = scipy.sparse.coo_matrix((amplitudes, (positions, fIndices)), shape=(sequence.shape[0], D.shape[0]))
            coefficients = coefficients.tocsc()
            
            residual = residual.astype(np.float32)
            
        finally:
            # Remove temporary files
            os.remove(dictFilePath)
            os.remove(tableFilePath)
            os.remove(dataFilePath)

        if squeezeOutput:
            residual = np.squeeze(residual, axis=1)

        return coefficients, residual


class Atom(object):
    
    def __init__(self, position, index, coefficient, length):
        self.__dict__.update(position=position, index=index, coefficient=coefficient, length=length)

    def getPositionSpanIndices(self, sequenceLength=None):
        if np.mod(self.length, 2) == 0:
            # Even
            startIdx = self.position - (self.length/2-1)
        else:
            # Odd
            startIdx = self.position - (self.length/2)
        endIdx = self.position + (self.length/2)
        
        if sequenceLength is not None:
            startIdx = max(startIdx, 0)
            endIdx = min(endIdx, sequenceLength-1)
            
        return startIdx, endIdx

    def __str__(self):
        return 'Atom no.%d of length %d at position %d, c = %4.10f' % (self.index, self.length, self.position, self.coefficient)

    def __repr__(self):
        return self.__str__()

class ConvolutionalMatchingPursuit(SparseApproximator):

    def __init__(self, verbose=False):

        if verbose:
            plt.ion()
            self.fig = plt.figure(figsize=(12,8), facecolor='white', frameon=True)
        else:
            self.fig = None
        self.verbose = verbose

    def _updateDemo(self, atom, D, residual):
        
        logger.info('Matching pursuit: raw event is (t = %d, f = %d, c = %f)' % (atom.position, atom.index, atom.coefficient))
                    
        self.fig.clf()
        ax1 = self.fig.add_subplot(211)
        ax1.plot(residual, '-k', linewidth=2)
        
        basis = atom.coefficient*D[atom.index]
        startIdx, endIdx = atom.getPositionSpanIndices(residual.shape[0])
        basisTime = np.arange(startIdx, endIdx+1)
        ax1.plot(np.ones_like(basis) * basisTime[:,np.newaxis], basis, '-r', linewidth=1)
        ax1.set_title('Coefficient = %f' % (atom.coefficient))
        ax1.axvline(x=atom.position, color='gray', linestyle='--')
        ax1.set_xlim((atom.position - 2 * D.shape[1], atom.position + 2 * D.shape[1]))
        
        ax2 = self.fig.add_subplot(212)
        ax2.plot(residual, '-k', linewidth=2)
        ax2.axvline(x=atom.position, color='gray', linestyle='--')
        
        self.fig.canvas.draw()

    def _selectBestAtoms(self, innerProducts, filterWidth, nbBlocks=1, offset=False, nullCoeffThres=0.0, weights=None):
        
         # Calculate the score for selection
        if weights is None:
            scores = innerProducts
        else:
            assert len(weights) == innerProducts.shape[1]
            scores = innerProducts * weights[np.newaxis,:]
        
        if nbBlocks is None or nbBlocks > 1:
            # Calculate the number of blocks
            if nbBlocks is None:
                # If the number of blocks was not provided, set it automatically so that block sizes are of 4 times the length of the filters
                blockSize = 4 * filterWidth
            else:
                blockSize = int(np.floor(scores.shape[0] / float(nbBlocks)))
            # Make sure block size is even
            if np.mod(blockSize, 2) == 1:
                blockSize += 1
            nbBlocks = int(np.ceil(scores.shape[0] / float(blockSize)))
            
            # Calculate padding needed for even-sized blocks
            basePadEnd = nbBlocks*blockSize - scores.shape[0]
            if offset:
                # With offset
                padding = (blockSize/2, basePadEnd + blockSize/2)
                nbBlocks += 1
            else:
                # No offset
                padding = (0, basePadEnd)
            
            # Pad and split into multiple windows (blocks)
            scores = np.pad(scores, [padding,] + [(0,0) for _ in range(scores.ndim-1)], mode='constant')
            windows = np.stack(np.split(scores, nbBlocks, axis=0))
            
            # Get maximum activation inside each block
            tRel, fIdx = np.unravel_index(np.argmax(np.abs(windows.reshape((windows.shape[0], windows.shape[1]*windows.shape[2]))), axis=1),
                                       dims=(windows.shape[1], windows.shape[2]))
            t = tRel + np.arange(0, nbBlocks*blockSize, step=blockSize, dtype=np.int) - padding[0]
            
            # Remove activations outside the valid range (e.g. because of padding)
            indices = np.where((t >= 0) & (t <= innerProducts.shape[0]-1))[0]
            nbInvalidRange = len(t) - len(indices)
            t, fIdx = t[indices], fIdx[indices]
            logger.debug('Number of activations with invalid range removed during selection: %d' % (nbInvalidRange))
            
            # Remove activations that have null coefficients (e.g. because of padding)
            coefficients = innerProducts[t, fIdx]
            nullMask = np.where(np.abs(coefficients) > nullCoeffThres)
            t, fIdx, coefficients = t[nullMask], fIdx[nullMask], coefficients[nullMask]
            
            # Remove activations that would cause interference (too close across block boundaries), always keep the first activation
            tDiff = t[1:] - t[:-1]
            indices = np.where(tDiff >= filterWidth)[0]
            if len(indices) > 0:
                indices = np.concatenate(([0], indices + 1))
                nbInterferences = len(t) - len(indices)
                t, fIdx, coefficients = t[indices], fIdx[indices], coefficients[indices]
                logger.debug('Number of interfering activations removed during selection: %d' % (nbInterferences))
            
            # Sort activations by absolute amplitude of coefficients
            indices = np.argsort(np.abs(coefficients))[::-1]
            nbNulls = len(t) - len(indices)
            t, fIdx, coefficients = t[indices], fIdx[indices], coefficients[indices]
            logger.debug('Number of null activations removed during selection: %d' % (nbNulls))
            
        else:
            # Find maximum across the whole activations
            t, fIdx = np.unravel_index(np.argmax(np.abs(scores)), scores.shape)
            t = np.stack((t,))
            fIdx = np.stack((fIdx,))
            coefficients = innerProducts[t, fIdx]
            
            # Remove activations that have null coefficients (e.g. because of padding)
            coefficients = innerProducts[t, fIdx]
            nullMask = np.where(np.abs(coefficients) > nullCoeffThres)
            t, fIdx, coefficients = t[nullMask], fIdx[nullMask], coefficients[nullMask]
            
        assert np.all(t >= 0) and np.all(t < innerProducts.shape[0])
        assert np.all(fIdx >= 0) and np.all(fIdx < innerProducts.shape[1])
        
        # Create the list of atom instances
        atoms = [Atom(p, f, c, filterWidth) for p, f, c in zip(t, fIdx, coefficients)]
        return atoms
        
    def _updateCoefficients(self, coefficients, atoms, replace=True):
        
        positions = [atom.position for atom in atoms]
        indices = [atom.index for atom in atoms]
        coefficientsAtom = np.array([atom.coefficient for atom in atoms], dtype=coefficients.dtype)
        if replace:
            coefficients[positions, indices] = coefficientsAtom
        else:
            coefficients[positions, indices] += coefficientsAtom
            
        return coefficients
    
    def _updateResidual(self, residual, energyResidual, atoms, D, eps=1e-16):

        residualCopy = np.copy(residual)

        energyLoss = 0.0
        for atom in atoms:
            # Update the residual by removing the contribution of the selected filter
            # NOTE: negate the coefficient to actually perform a overlap-remove operation.
            localEnergyBefore = np.sum(np.square(peek(residual, atom.length, atom.position)))
            overlapAdd(residual, -atom.coefficient*D[atom.index], atom.position, copy=False)
            localEnergyAfter = np.sum(np.square(peek(residual, atom.length, atom.position)))
            energyLoss += (localEnergyBefore - localEnergyAfter)
        
        if energyLoss < 0.0 and np.abs(energyLoss) > eps:
            logger.warn('Residual energy (%f) increased by %4.18f' % (energyResidual, -energyLoss))
            
#             plt.ioff()
#             for atom in atoms:
#                 self.fig = plt.figure(figsize=(12,8), facecolor='white', frameon=True)
#                 self._updateDemo(atom, D, residualCopy)
#                 plt.show()
                
            if self.verbose:
                plt.ioff()
                plt.show()
            #assert energyLoss >= 0.0
        energyResidual -= energyLoss
        
        return residual, energyResidual
        
    def _updateInnerProducts(self, innerProducts, residual, atoms, D):
        
        # TODO: optimize by removing loop since all atoms should be close together (i.e. they have common support)
        for atom in atoms:
        
            # Update the inner products
            # First, calculate the span of the residual that needs to be convolved again, 
            # and the padding required if at the beginning or end of the residual
            width = atom.length
            padStart = 0
            if np.mod(width, 2) == 0:
                # Even width
                tstart = atom.position-width/2+1-(width-1)
            else:
                # Odd width
                tstart = atom.position-width/2-(width-1)
            startIdx = max(0,tstart)
            if tstart < 0:
                padStart = -tstart
                    
            tend = atom.position+width/2+(width-1)
            endIdx = min(residual.shape[0]-1, tend)
            padEnd = 0
            if tend > residual.shape[0]-1:
                padEnd = tend - (residual.shape[0]-1)
            assert endIdx - startIdx >= 0
            assert padStart >= 0 and padEnd >= 0
             
            paddedResidual = np.pad(residual[startIdx:endIdx+1], [(padStart, padEnd)] + [(0,0) for _ in range(residual.ndim-1)], mode='reflect')
            localInnerProducts = convolve1d(paddedResidual, D, padding='valid')
            assert localInnerProducts.shape[0] == width + (width-1)
            overlapReplace(innerProducts, localInnerProducts, atom.position, copy=False)
        
        return innerProducts
        
    def computeCoefficients(self, sequence, D, nbNonzeroCoefs=None, toleranceResidualScale=None, toleranceSnr=None, nbBlocks=1, alpha=0.0, minCoefficients=1e-16, weights=None, stopCondition=None):
        assert sequence.ndim == 1 or sequence.ndim == 2
        assert D.ndim == 2 or D.ndim == 3

        eps = np.finfo(D.dtype).eps

        if sequence.ndim == 1:
            squeezeOutput = True
            sequence = sequence[:,np.newaxis]
        else:
            squeezeOutput = False
            
        if D.ndim == 2:
            squeezeOutput = True
            D = D[:,:,np.newaxis]

        # Initialize the residual and sparse coefficients
        energySignal = np.sum(np.square(sequence))
        residual = np.copy(sequence)
        energyResidual = energySignal
        
        coefficients = scipy.sparse.lil_matrix((sequence.shape[0], D.shape[0]))
        
        # Convolve the input signal once, then locally recompute the affected projections when the residual is changed. 
        innerProducts = convolve1d(residual, D, padding='same')
                
        # Loop until convergence or if any stopping criteria is met
        nbIterations = 0
        nnz = 0
        nbDuplicates = 0
        offset = False
        converged = False
        nbSelections = 0
        while not converged:

            # Adaptive selection: rejection if coefficients are less that alpha times the maximum.
            nullCoeffThres = alpha * np.max(np.abs(innerProducts))
            if minCoefficients is not None:
                nullCoeffThres = max(minCoefficients, nullCoeffThres)
            atoms = self._selectBestAtoms(innerProducts, nbBlocks=nbBlocks, filterWidth=D.shape[1], offset=offset, nullCoeffThres=nullCoeffThres, weights=weights)
            for atom in atoms:
        
                if self.verbose:
                    self._updateDemo(atom, D, residual)
                
                if np.abs(coefficients[atom.position, atom.index]) > 0.0:
                    # Count duplicate coefficients if already existing
                    nbDuplicates += 1
                elif np.abs(atom.coefficient) > 0.0:
                    # Count non-zero coefficients
                    nnz += 1
                
                # Update the sparse coefficients
                coefficients = self._updateCoefficients(coefficients, [atom,], replace=False)
                
                # Update the residual by removing the contribution of the selected atoms
                residual, energyResidual = self._updateResidual(residual, energyResidual, [atom,], D)
                
                # Update the inner products
                innerProducts = self._updateInnerProducts(innerProducts, residual, [atom,], D)
                
                nbIterations += 1
                
                # Print information about current iteration
                if energyResidual < eps:
                    # Avoid numerical errors by setting to infinity
                    snr = np.inf
                    logger.debug('Perfect reconstruction reached: considering convergence is achieved')
                    converged = True
                    break
                else:
                    snr = 10.0*np.log10(energySignal/energyResidual)
                        
                # Check stopping criteria (fast)
                if nbNonzeroCoefs is not None and nnz >= nbNonzeroCoefs:
                    logger.debug('Tolerance for number of non-zero coefficients reached')
                    converged = True
                    break
                if toleranceSnr is not None and snr >= toleranceSnr:
                    logger.debug('Tolerance for signal-to-noise ratio reached')
                    converged = True
                    break
                
            # Check stopping criteria (slow)
            residualScale = np.max(np.abs(residual))
            if toleranceResidualScale is not None and residualScale <= toleranceResidualScale:
                logger.debug('Tolerance for residual scale (absolute value) reached')
                converged = True

            if len(atoms) == 0:
                # This means all coefficients are null
                logger.warn('Selection returned empty set: considering convergence is achieved')
                converged = True
                
            if stopCondition is not None:
                if stopCondition(coefficients):
                    logger.warn('Custom stop condition reached: considering convergence is achieved')
                    converged = True
                
            nbSelections += 1

            # Toggle offset switch
            offset = not offset
            
            # Print information
            logger.debug('SNR of %f dB achieved after %d selection iterations' % (snr, nbSelections))
            logger.debug('Number of selection: %d' % (len(atoms)))
            logger.debug('Number of non-zero coefficients: %d' % (nnz))
            logger.debug('Number of duplicate coefficients: %d' % (nbDuplicates))

        if minCoefficients is not None:
            # Clip small coefficients to zero
            clippedCoefficients = scipy.sparse.lil_matrix((sequence.shape[0], D.shape[0]))
            cx = coefficients.tocoo()
            nullMask = np.where(np.abs(cx.data) >= minCoefficients)
            clippedCoefficients[cx.row[nullMask],  cx.col[nullMask]] = cx.data[nullMask]
            coefficients = clippedCoefficients

        # Convert to compressed-column sparse matrix format
        coefficients = coefficients.tocsc()
        coefficients.eliminate_zeros()

        if squeezeOutput:
            residual = np.squeeze(residual, axis=1)

        return coefficients, residual

# TODO: implement interference adaptation, see:
# http://ieeexplore.ieee.org/document/5352303/

class LoCOMP(ConvolutionalMatchingPursuit):
    """
    Low-complexity Orthogonal Matching Pursuit (LoCOMP)
    From: A low complexity Orthogonal Matching Pursuit for sparse signal approximation with shift-invariant dictionaries
    http://ieeexplore.ieee.org/document/4960366/
    
    For information about OMP, see:
    https://www.mathworks.com/help/wavelet/ug/matching-pursuit-algorithms.html
    """
    
    def __init__(self, verbose=False):
        super(LoCOMP, self).__init__(verbose)
    
    def _precomputeGramMatrixForShifts(self, D):
        """
        Compute the Gram matrix between dictionary bases for all time shifts, depending on the filter width.
        """
        G = []
        filterWidth = D.shape[1]
        for t in range(-filterWidth+1, filterWidth):
            if t < 0:
                shiftedD = np.pad(D[:, :filterWidth+t], [(0,0), (0,-t)] + [(0,0) for _ in range(D.ndim-2)], mode='constant')
            elif t >= 0:
                shiftedD = np.pad(D[:, t:], [(0,0), (t,0)] + [(0,0) for _ in range(D.ndim-2)], mode='constant')
            Gt = np.dot(shiftedD.reshape(D.shape[0], -1), D.reshape(D.shape[0], -1).T)
            G.append(Gt)
        
        G = np.stack(G, axis=2)
        return G

    def _findCommonSupportAtoms(self, atom, coefficients, D):
        
        # TODO: is it more efficient to use CSC format rather than LIL for column indexing?
        #coefficients = coefficients.tocsc()
        
        sequenceLength = coefficients.shape[0]
        atomLength = D.shape[1]
        startIdx, endIdx = atom.getPositionSpanIndices(sequenceLength)
        startIdx = max(startIdx - atomLength/2, 0)
        if np.mod(atomLength, 2) == 0:
            # Even
            endIdx = min(endIdx + atomLength/2-1, sequenceLength)
        else:
            endIdx = min(endIdx + atomLength/2, sequenceLength)
        subspace = coefficients[startIdx:endIdx+1,:].tocoo()
        
        # NOTE: make sure we dont add the reference atom, if it already exists in the set of coefficient
        atoms = [Atom(startIdx+position, index, coefficient, atomLength) for position, index, coefficient in zip(subspace.row, subspace.col, subspace.data) if position != atom.position and index != atom.index]
        return atoms
    
    def _getDictionaryFromSupportAtoms(self, sequence, atoms, D):
        
        minStartIdx = np.Inf
        maxEndIdx = 0
        for atom in atoms:
            startIdx, endIdx = atom.getPositionSpanIndices(sequence.shape[0])
            if startIdx < minStartIdx:
                minStartIdx = startIdx
            if endIdx > maxEndIdx:
                maxEndIdx = endIdx
        supportLength = maxEndIdx - minStartIdx + 1
        
        Dsup = []
        for atom  in atoms:
            signal = np.zeros((supportLength,) + sequence.shape[1:], dtype=D.dtype)
            overlapAdd(signal, D[atom.index], atom.position - minStartIdx, copy=False)
            Dsup.append(signal)
        Dsup = np.stack(Dsup)
        sequenceSup = sequence[minStartIdx:maxEndIdx+1]
        
        return Dsup, sequenceSup
    
    def computeCoefficients(self, sequence, D, nbNonzeroCoefs=None, toleranceResidualScale=None, toleranceSnr=None, nbBlocks=1, alpha=0.0, minCoefficients=1e-16, weights=None, stopCondition=None, G=None):
        assert sequence.ndim == 1 or sequence.ndim == 2
        assert D.ndim == 2 or D.ndim == 3

        eps = np.finfo(D.dtype).eps

        if sequence.ndim == 1:
            squeezeOutput = True
            sequence = sequence[:,np.newaxis]
        else:
            squeezeOutput = False
            
        if D.ndim == 2:
            squeezeOutput = True
            D = D[:,:,np.newaxis]

        if G is None:
            G = self._precomputeGramMatrixForShifts(D)

        # Initialize the residual and sparse coefficients
        energySignal = np.sum(np.square(sequence))
        residual = np.copy(sequence)
        energyResidual = energySignal
        
        coefficients = scipy.sparse.lil_matrix((sequence.shape[0], D.shape[0]))
        
        # Convolve the input signal once, then locally recompute the affected projections when the residual is changed. 
        innerProducts = convolve1d(residual, D, padding='same')
                
        # Loop until convergence or if any stopping criteria is met
        nbIterations = 0
        offset = False
        converged = False
        nbSelections = 0
        while not converged:

            # Adaptive selection: rejection if coefficients are less that alpha times the maximum.
            nullCoeffThres = alpha * np.max(np.abs(innerProducts))
            if minCoefficients is not None:
                nullCoeffThres = max(minCoefficients, nullCoeffThres)
            atoms = self._selectBestAtoms(innerProducts, nbBlocks=nbBlocks, filterWidth=D.shape[1], offset=offset, nullCoeffThres=nullCoeffThres, weights=weights)
            for atom in atoms:
        
                if self.verbose:
                    self._updateDemo(atom, D, residual)
        
                if coefficients[atom.position, atom.index] != 0.0:
                    logger.warn('Redundant atom selected: %s' % (str(atom)))
                    if self.verbose:
                        plt.ioff()
                        plt.show()
                    #assert coefficients[atom.position, atom.index] == 0.0
                    
                lastEnergyResidual = energyResidual
                commonSupportAtoms = self._findCommonSupportAtoms(atom, coefficients, D)
                if len(commonSupportAtoms) > 0:
                    # Common support with other atoms, so calculate the orthogonal projection
                    commonSupportAtoms = [atom,] + commonSupportAtoms
                    Dsup, residualSup =  self._getDictionaryFromSupportAtoms(residual, commonSupportAtoms, D)
                
                    # Calculate the orthogonal projection of the dictionary on the residual
                    # NOTE: use the Moore-Penrose pseudoinverse method
                    DsupF = Dsup.reshape((Dsup.shape[0], -1))
                    #Gsup = np.dot(np.linalg.inv(np.dot(DsupF, DsupF.T)), DsupF)
                    Gsup = np.linalg.pinv(DsupF).T
                    coefficientsSup = np.dot(Gsup, residualSup.flatten()).flatten()
                    
                    innerProductsSup = convolve1d(residualSup, D, padding='same')
#                     print innerProductsSup.shape, Dsup.shape
#                     for i, commonSupportAtom in enumerate(commonSupportAtoms):
#                         print innerProducts[commonSupportAtom.position, commonSupportAtom.index], np.dot(DsupF[i], residualSup.flatten())
                    #print Dsup[0].flatten()
                    #print D[commonSupportAtoms[0].index].flatten()
                    #print coefficientsSup
                
                    # Update the sparse coefficients
                    coefficientsSupOrig = np.array([coefficients[commonSupportAtom.position, commonSupportAtom.index] for commonSupportAtom in commonSupportAtoms], dtype=coefficients.dtype)
                    for commonSupportAtom, coefficientSup in zip(commonSupportAtoms, coefficientsSup):
                        commonSupportAtom.coefficient = coefficientSup
                    coefficients = self._updateCoefficients(coefficients, commonSupportAtoms, replace=False)
                
                    # Update the residual by removing the contribution of the selected atoms
                    residual, energyResidual = self._updateResidual(residual, energyResidual, commonSupportAtoms, D, eps)
                else:
                    # No common support with other atoms
                    commonSupportAtoms = [atom,]
                    
                    # Update the sparse coefficients
                    coefficients = self._updateCoefficients(coefficients, commonSupportAtoms, replace=False)
                    
                    # Update the residual by removing the contribution of the selected atoms
                    residual, energyResidual = self._updateResidual(residual, energyResidual, commonSupportAtoms, D, eps)
                
                # Update the inner products
                innerProducts = self._updateInnerProducts(innerProducts, residual, commonSupportAtoms, D)
                
                nbIterations += 1
                
                # Print information about current iteration
                if energyResidual < eps:
                    # Avoid numerical errors by setting to infinity
                    snr = np.inf
                    logger.debug('Perfect reconstruction reached: considering convergence is achieved')
                    converged = True
                    break
                else:
                    snr = 10.0*np.log10(energySignal/energyResidual)
 
                # Check stopping criteria (fast)
                if nbNonzeroCoefs is not None and coefficients.nnz >= nbNonzeroCoefs:
                    logger.debug('Tolerance for number of non-zero coefficients reached')
                    converged = True
                    break
                if toleranceSnr is not None and snr >= toleranceSnr:
                    logger.debug('Tolerance for signal-to-noise ratio reached')
                    converged = True
                    break
                
                deltaEnergyResidual = np.abs(lastEnergyResidual - energyResidual)
                if deltaEnergyResidual < eps:
                    logger.warn('Residual energy is no more reduced: considering convergence is achieved')
                    converged = True
                    break
                lastEnergyResidual = energyResidual
                
            # Check stopping criteria (slow)
            residualScale = np.max(np.abs(residual))
            if toleranceResidualScale is not None and residualScale <= toleranceResidualScale:
                logger.debug('Tolerance for residual scale (absolute value) reached')
                converged = True

            if len(atoms) == 0:
                # This means all coefficients are null
                logger.warn('Selection returned empty set: considering convergence is achieved')
                converged = True
                
            if stopCondition is not None:
                if stopCondition(coefficients):
                    logger.warn('Custom stop condition reached: considering convergence is achieved')
                    converged = True
                
            nbSelections += 1

            # Toggle offset switch
            offset = not offset
            
            # Print information
            logger.debug('SNR of %f dB achieved after %d selection iterations' % (snr, nbSelections))
            logger.debug('Number of selection: %d' % (len(atoms)))
            logger.debug('Number of non-zero coefficients: %d' % (coefficients.nnz))

        if minCoefficients is not None:
            # Clip small coefficients to zero
            clippedCoefficients = scipy.sparse.lil_matrix((sequence.shape[0], D.shape[0]))
            cx = coefficients.tocoo()
            nullMask = np.where(np.abs(cx.data) >= minCoefficients)
            clippedCoefficients[cx.row[nullMask],  cx.col[nullMask]] = cx.data[nullMask]
            coefficients = clippedCoefficients

        # Convert to compressed-column sparse matrix format
        coefficients = coefficients.tocsc()
        coefficients.eliminate_zeros()

        if squeezeOutput:
            residual = np.squeeze(residual, axis=1)

        return coefficients, residual

class HierarchicalConvolutionalMatchingPursuit(SparseApproximator):

    def __init__(self, method='locomp'):
        self.method = method

    def _forwardPhase(self, sequence, multilevelDict, toleranceSnr=None, nbBlocks=1, alpha=0.5, singletonWeight=0.5, stopCondition=None):
        
        # Loop over all levels
        input = sequence
        coefficients = []
        for level in range(multilevelDict.getNbLevels()):
        
            if toleranceSnr is not None and isinstance(toleranceSnr, collections.Iterable):
                targetSnr = toleranceSnr[level]
            else:
                targetSnr = toleranceSnr
        
            # Get dictionary at current level
            D = multilevelDict.getRawDictionary(level)
        
            # Calculate the weights (reduce the score of the singletons)
            nbSingletons = D.shape[0] - multilevelDict.countsNoSingletons[level]
            weights = np.ones((D.shape[0],), dtype=D.dtype)
            weights[:nbSingletons] = singletonWeight
        
            energySignal = np.sum(np.square(sequence))
            def stopInputSnrTolerance(c):
                residual = self._calculateResidual(sequence, coefficients + [c,], multilevelDict)
                energyResidual = np.sum(np.square(residual))
                
                if energyResidual == 0.0:
                    # Avoid numerical errors by setting to infinity
                    snr = np.inf
                else:
                    snr = 10.0*np.log10(energySignal/energyResidual)
                    
                if snr >= targetSnr:
                    condition = True
                else:
                    condition = False
                return condition
        
            # Instanciate a new coder for current level and encode
            if self.method == 'mptk-mp':
                cmp = MptkConvolutionalMatchingPursuit(method='mp')
                levelCoder = ConvolutionalSparseCoder(D, cmp)
                levelCoefficients, residual = levelCoder.encode(input, toleranceSnr=targetSnr)
            elif self.method == 'mptk-cmp':
                cmp = MptkConvolutionalMatchingPursuit(method='cmp')
                levelCoder = ConvolutionalSparseCoder(D, cmp)
                levelCoefficients, residual = levelCoder.encode(input, toleranceSnr=targetSnr)
            elif self.method == 'locomp':
                cmp = LoCOMP()
                levelCoder = ConvolutionalSparseCoder(D, cmp)
                levelCoefficients, residual = levelCoder.encode(input, toleranceSnr=targetSnr, nbBlocks=nbBlocks, alpha=alpha, weights=weights)
            elif self.method == 'cmp':
                cmp = ConvolutionalMatchingPursuit()
                levelCoder = ConvolutionalSparseCoder(D, cmp)
                levelCoefficients, residual = levelCoder.encode(input, toleranceSnr=targetSnr, nbBlocks=nbBlocks, alpha=alpha, weights=weights)
            else:
                raise Exception('Unsupported sparse coding method: %s' % (self.method))
            
            input = levelCoefficients.todense()
            coefficients.append(levelCoefficients)
        
        return coefficients

    def _forwardPhaseFromLevel(self, sequence, coefficients, multilevelDict, toleranceSnr=None, nbBlocks=1, alpha=0.5, singletonWeight=0.5, stopCondition=None):
        
        # Loop over all remaining levels, starting from the last level in the provided coefficients
        fromLevel = len(coefficients)
        input = coefficients[-1].todense()
        for level in range(fromLevel, multilevelDict.getNbLevels()):
        
            if toleranceSnr is not None and isinstance(toleranceSnr, collections.Iterable):
                targetSnr = toleranceSnr[level]
            else:
                targetSnr = toleranceSnr
        
            # Get dictionary at current level
            D = multilevelDict.getRawDictionary(level)
        
            # Calculate the weights (reduce the score of the singletons)
            nbSingletons = D.shape[0] - multilevelDict.countsNoSingletons[level]
            weights = np.ones((D.shape[0],), dtype=D.dtype)
            weights[:nbSingletons] = singletonWeight
        
            energySignal = np.sum(np.square(sequence))
            def stopInputSnrTolerance(c):
                residual = self._calculateResidual(sequence, coefficients + [c,], multilevelDict)
                energyResidual = np.sum(np.square(residual))
                
                if energyResidual == 0.0:
                    # Avoid numerical errors by setting to infinity
                    snr = np.inf
                else:
                    snr = 10.0*np.log10(energySignal/energyResidual)
                    
                if snr >= targetSnr:
                    condition = True
                else:
                    condition = False
                return condition
        
            # Instanciate a new coder for current level and encode
            if self.method == 'mptk-mp':
                cmp = MptkConvolutionalMatchingPursuit(method='mp')
                levelCoder = ConvolutionalSparseCoder(D, cmp)
                levelCoefficients, residual = levelCoder.encode(input, toleranceSnr=targetSnr)
            elif self.method == 'mptk-cmp':
                cmp = MptkConvolutionalMatchingPursuit(method='cmp')
                levelCoder = ConvolutionalSparseCoder(D, cmp)
                levelCoefficients, residual = levelCoder.encode(input, toleranceSnr=targetSnr)
            elif self.method == 'locomp':
                cmp = LoCOMP()
                levelCoder = ConvolutionalSparseCoder(D, cmp)
                levelCoefficients, residual = levelCoder.encode(input, toleranceSnr=targetSnr, nbBlocks=nbBlocks, alpha=alpha, weights=weights)
            elif self.method == 'cmp':
                cmp = ConvolutionalMatchingPursuit()
                levelCoder = ConvolutionalSparseCoder(D, cmp)
                levelCoefficients, residual = levelCoder.encode(input, toleranceSnr=targetSnr, nbBlocks=nbBlocks, alpha=alpha, weights=weights)
            else:
                raise Exception('Unsupported sparse coding method: %s' % (self.method))
            
            input = levelCoefficients.todense()
            coefficients.append(levelCoefficients)
        
        return coefficients

    def convertToDistributedCoefficients(self, coefficients):
        
        lastLevelCoefficients = coefficients[-1].copy()
        if scipy.sparse.issparse(lastLevelCoefficients):
            # NOTE: convert to compressed-column format for efficient slicing and addition
            lastLevelCoefficients = lastLevelCoefficients.tocsc()
        
        # Loop over all levels
        newCoefficients = []
        for level in range(len(coefficients)):
            
            if level < len(coefficients) - 1:
                levelCoefficients = coefficients[level]
                nbFeatures = levelCoefficients.shape[1]
                
                if scipy.sparse.issparse(levelCoefficients):
                    
                    # Extract the related singleton coefficients from the last layer, then set to zero
                    levelCoefficients = lastLevelCoefficients[:, :nbFeatures]
                    lastLevelCoefficients = scipy.sparse.hstack((scipy.sparse.csc_matrix((lastLevelCoefficients.shape[0], nbFeatures), dtype=lastLevelCoefficients.dtype),
                                                                 lastLevelCoefficients[:, nbFeatures:]))
                    
                    # Remove any zero entries in the sparse matrix
                    levelCoefficients.eliminate_zeros()
                else:
                    # Extract the related singleton coefficients from the last layer, then set to zero
                    levelCoefficients = lastLevelCoefficients[:, :nbFeatures]
                    lastLevelCoefficients[:, :nbFeaturesLevel] = 0.0
            else:
                # Last layer
                levelCoefficients = lastLevelCoefficients
            
            newCoefficients.append(levelCoefficients)
        
        # NOTE: the total number of coefficients should not change compared to the last layer
        assert len(newCoefficients) == len(coefficients)
        assert np.sum([c.nnz for c in newCoefficients]) == coefficients[-1].nnz
            
        return newCoefficients

    def _calculateResidual(self, sequence, coefficients, multilevelDict):
        
        # Get the number of features at input level
        baseDict = multilevelDict.getBaseDictionary()
        if baseDict.ndim == 2:
            reconstruction = np.zeros((coefficients[0].shape[0],), dtype=coefficients[0].dtype)
        else:
            reconstruction = np.zeros((coefficients[0].shape[0], baseDict.shape[-1]), dtype=coefficients[0].dtype)
            
        # Calculate residual based on the reconstruction
        representations = multilevelDict.getMultiscaleDictionaries()
        for level in range(multilevelDict.getNbLevels()):
            reconstruction += reconstructSignal(coefficients[level], representations[level])
        residual = sequence - reconstruction
        
        return residual

    def _postprocessCoefficients(self, coefficients, multilevelDict, returnDistributed=True):

        # Remove redundancy across layers
        if returnDistributed:
            coefficients = self.convertToDistributedCoefficients(coefficients)
        else:
             # Loop over all levels
            newCoefficients = []
            for level in range(multilevelDict.getNbLevels()):
                levelCoefficients = coefficients[level]
                
                # Keep last-level only
                if level < multilevelDict.getNbLevels() - 1:
                    if scipy.sparse.issparse(levelCoefficients):
                        levelCoefficients = scipy.sparse.csc_matrix(levelCoefficients.shape, dtype=levelCoefficients.dtype)
                    else:
                        levelCoefficients = np.zeros_like(levelCoefficients)
                
                newCoefficients.append(levelCoefficients)
            coefficients = newCoefficients
        
        return coefficients
    
    def computeCoefficients(self, sequence, multilevelDict, nbNonzeroCoefs=None, toleranceResidualScale=None, toleranceSnr=None, nbBlocks=1, alpha=0.5, minCoefficients=None, singletonWeight=0.5, returnDistributed=True, stopCondition=None):
        assert isinstance(multilevelDict, MultilevelDictionary)
        
        # Encode signal bottom-up through all layers
        coefficients = self._forwardPhase(sequence, multilevelDict, toleranceSnr, nbBlocks, alpha, singletonWeight, stopCondition)
        coefficients = self._postprocessCoefficients(coefficients, multilevelDict, returnDistributed)
        residual = self._calculateResidual(sequence, coefficients, multilevelDict)
        return coefficients, residual
    
    def computeCoefficientsFromLevel(self, sequence, coefficients, multilevelDict, nbNonzeroCoefs=None, toleranceResidualScale=None, toleranceSnr=None, nbBlocks=1, alpha=0.5, minCoefficients=None, singletonWeight=0.5, stopCondition=None, returnDistributed=True):
        assert isinstance(multilevelDict, MultilevelDictionary)

        # FIXME: not sure why deep copy is needed here, maybe because this is a list instance
        coefficients = copy.deepcopy(coefficients)

        # Encode signal bottom-up through all layers
        coefficients = self._forwardPhaseFromLevel(sequence, coefficients, multilevelDict, toleranceSnr, nbBlocks, alpha, singletonWeight, stopCondition)
        coefficients = self._postprocessCoefficients(coefficients, multilevelDict, returnDistributed)
        return coefficients
    
class ConvolutionalSparseCoder(object):
 
    def __init__(self, D, approximator):
        assert D.ndim == 2 or D.ndim == 3
        self.D = D
        self.approximator = approximator
 
    def encode(self, X, *args, **kwargs):
        assert X.ndim == 1 or X.ndim == 2
        return self.approximator.computeCoefficients(X, self.D, *args, **kwargs)
 
    def reconstruct(self, coefficients):
        assert coefficients.ndim == 1 or coefficients.ndim == 2
        return reconstructSignal(coefficients, self.D)
    
class HierarchicalConvolutionalSparseCoder(object):

    def __init__(self, multilevelDict, approximator):
        assert isinstance(multilevelDict, MultilevelDictionary)
        
        if not multilevelDict.hasSingletonBases:
            multilevelDict = multilevelDict.withSingletonBases()
        
        self.multilevelDict = multilevelDict
        self.approximator = approximator
 
    def encode(self, sequence, *args, **kwargs):
        assert sequence.ndim == 1 or sequence.ndim == 2
        return self.approximator.computeCoefficients(sequence, self.multilevelDict, *args, **kwargs)

    def encodeFromLevel(self, sequence, coefficients, *args, **kwargs):
        assert len(coefficients) > 0
        return self.approximator.computeCoefficientsFromLevel(sequence, coefficients, self.multilevelDict, *args, **kwargs)
 
    def reconstruct(self, coefficients):
        assert len(coefficients) > 0
        
        # Get the number of features at input level
        baseDict = self.multilevelDict.getBaseDictionary()
        if baseDict.ndim == 2:
            signal = np.zeros((coefficients[0].shape[0],), dtype=coefficients[0].dtype)
        else:
            signal = np.zeros((coefficients[0].shape[0], baseDict.shape[-1]), dtype=coefficients[0].dtype)
            
        # Calculate residual based on the reconstruction
        representations = self.multilevelDict.getMultiscaleDictionaries()
        for level in range(self.multilevelDict.getNbLevels()):
            signal += reconstructSignal(coefficients[level], representations[level])
        
        return signal
