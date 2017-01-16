
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
import unittest
import scipy
import scipy.sparse
import numpy as np

from hsc.modeling import ConvolutionalMatchingPursuit, ConvolutionalSparseCoder, ConvolutionalDictionaryLearner, extractRandomWindows, convolve1d, convolve1d_batch, extractWindows
from hsc.utils import normalize, overlapAdd

class TestConvolutionalDictionaryLearner(unittest.TestCase):

    def test_train_samples(self):
        sequence = np.random.random(size=(256,))
        cdl = ConvolutionalDictionaryLearner(k=16, windowSize=5, algorithm='samples')
        D = cdl.train(sequence)
        self.assertTrue(np.array_equal(D.shape, [16,5]))

    def test_train_kmean(self):
        
        for initMethod in ['noise', 'random_samples']:
            sequence = np.random.random(size=(256,))
            cdl = ConvolutionalDictionaryLearner(k=16, windowSize=5, algorithm='kmean')
            D = cdl.train(sequence, nbRandomWindows=32, maxIterations=100, tolerance=0.0, initMethod=initMethod)
            self.assertTrue(np.array_equal(D.shape, [16,5]))
            
        for resetMethod in ['noise', 'random_samples', 'random_samples_average']:
            sequence = np.random.random(size=(256,))
            cdl = ConvolutionalDictionaryLearner(k=16, windowSize=5, algorithm='kmean')
            D = cdl.train(sequence, nbRandomWindows=32, maxIterations=100, tolerance=0.0, resetMethod=resetMethod)
            self.assertTrue(np.array_equal(D.shape, [16,5]))

class TestConvolutionalMatchingPursuit(unittest.TestCase):

    def test_computeCoefficients(self):
        
        nbNonzeroCoefs = 4
        for dsize in [1,2,3]:
            for dwidth in [3,5,6]:
                cmp = ConvolutionalMatchingPursuit()
                sequence = np.random.random(size=(16,))
                D = normalize(np.random.random(size=(dsize,dwidth)), axis=1)
                coefficients, residual = cmp.computeCoefficients(sequence, D, nbNonzeroCoefs=nbNonzeroCoefs)
                self.assertTrue(coefficients.nnz == nbNonzeroCoefs)
                self.assertTrue(np.sum(np.square(residual)) < np.sum(np.square(sequence)))
        
        nbNonzeroCoefs = 16
        cmp = ConvolutionalMatchingPursuit()
        sequence = np.random.random(size=(64,2))
        D = normalize(np.random.random(size=(16,15,2)), axis=(1,2))
        coefficients, residual = cmp.computeCoefficients(sequence, D, nbNonzeroCoefs=nbNonzeroCoefs)
        self.assertTrue(coefficients.nnz == nbNonzeroCoefs)
        self.assertTrue(np.sum(np.square(residual)) < np.sum(np.square(sequence)))

    def test_doSelection(self):
        
        filterWidth = 5
        nbBlocks = 4
        cmp = ConvolutionalMatchingPursuit()
        innerProducts = np.arange(256).reshape((64,4)).astype(np.float)
        innerProducts[-1] = innerProducts[-1][::-1]
        t, fIdx = cmp._doSelection(innerProducts, filterWidth, nbBlocks, offset=False)
        self.assertTrue(len(t) == nbBlocks)
        self.assertTrue(len(fIdx) == nbBlocks)
        self.assertTrue(np.array_equal(t,[63,47,31,15]))
        self.assertTrue(np.array_equal(fIdx,[0,3,3,3]))
        self.assertTrue(np.allclose(np.abs(innerProducts[t[0], fIdx[0]]), np.abs(np.max(innerProducts))))
        t, fIdx = cmp._doSelection(innerProducts, filterWidth, nbBlocks, offset=True)
        self.assertTrue(len(t) == nbBlocks+1)
        self.assertTrue(len(fIdx) == nbBlocks+1)
        self.assertTrue(np.array_equal(t,[63,55,39,23,7]))
        self.assertTrue(np.array_equal(fIdx,[0,3,3,3,3]))
        self.assertTrue(np.allclose(np.abs(innerProducts[t[0], fIdx[0]]), np.abs(np.max(innerProducts))))
        
        filterWidth = 3
        nbBlocks = None
        cmp = ConvolutionalMatchingPursuit()
        innerProducts = np.arange(256).reshape((64,4)).astype(np.float)
        innerProducts[-1] = innerProducts[-1][::-1]
        t, fIdx = cmp._doSelection(innerProducts, filterWidth, nbBlocks, offset=False)
        self.assertTrue(len(t) == 6)
        self.assertTrue(len(fIdx) == 6)
        self.assertTrue(np.array_equal(t,[63,59,47,35,23,11]))
        self.assertTrue(np.array_equal(fIdx,[0,3,3,3,3,3]))
        self.assertTrue(np.allclose(np.abs(innerProducts[t[0], fIdx[0]]), np.abs(np.max(innerProducts))))
        
        filterWidth = 5
        nbBlocks = 5
        cmp = ConvolutionalMatchingPursuit()
        innerProducts = np.arange(256).reshape((64,4)).astype(np.float)
        innerProducts[-1] = innerProducts[-1][::-1]
        t, fIdx = cmp._doSelection(innerProducts, filterWidth, nbBlocks, offset=False)
        self.assertTrue(len(t) == nbBlocks)
        self.assertTrue(len(fIdx) == nbBlocks)
        self.assertTrue(np.array_equal(t,[59,47,35,23,11])) # Missing 63 because of interference
        self.assertTrue(np.array_equal(fIdx,[3,3,3,3,3]))
        self.assertTrue(not np.allclose(np.abs(innerProducts[t[0], fIdx[0]]), np.abs(np.max(innerProducts))))
        
class TestConvolutionalSparseCoder(unittest.TestCase):

    def test_encode(self):
        
        for tolerance in [0.5, 0.1, 0.001, 0.000001]:
            cmp = ConvolutionalMatchingPursuit()
            nbComponents = 32
            filterWidth = 9
            nbFeatures = 1
            D = normalize(np.random.random(size=(nbComponents, filterWidth, nbFeatures)), axis=(1,2))
            csc = ConvolutionalSparseCoder(D, approximator=cmp)
            
            sequence = np.random.random(size=(128,1))
            coefficients, residual = csc.encode(sequence, nbNonzeroCoefs=None, toleranceResidualScale=tolerance)
            self.assertTrue(scipy.sparse.issparse(coefficients))
            self.assertTrue(coefficients.nnz > 0)
            self.assertTrue(np.max(np.abs(residual)) < tolerance)
            
        for tolerance in [5, 10, 20, 50]:
            cmp = ConvolutionalMatchingPursuit()
            nbComponents = 32
            filterWidth = 9
            nbFeatures = 1
            D = normalize(np.random.random(size=(nbComponents, filterWidth, nbFeatures)), axis=(1,2))
            csc = ConvolutionalSparseCoder(D, approximator=cmp)
            
            sequence = np.random.random(size=(128,1))
            coefficients, residual = csc.encode(sequence, nbNonzeroCoefs=None, toleranceSnr=tolerance)
            snr = 10.0*np.log10(np.sum(np.square(sequence))/np.sum(np.square(residual)))
            self.assertTrue(scipy.sparse.issparse(coefficients))
            self.assertTrue(coefficients.nnz > 0)
            self.assertTrue(snr >= tolerance)
    
    def test_optimality(self):
        cmp = ConvolutionalMatchingPursuit()
        nbComponents = 4
        filterWidth = 32
        D = normalize(np.random.random(size=(nbComponents, filterWidth)), axis=1)
        csc = ConvolutionalSparseCoder(D, approximator=cmp)
        
        sequence = np.zeros((256,), dtype=np.float32)
        overlapAdd(sequence, D[0], t=32, copy=False)
        overlapAdd(sequence, D[3], t=48, copy=False)
        overlapAdd(sequence, 0.5*D[1], t=64, copy=False)
        overlapAdd(sequence, D[0], t=96, copy=False)
        overlapAdd(sequence, 0.75*D[2], t=128, copy=False)
        overlapAdd(sequence, 2.0*D[2], t=192, copy=False)
        
        coefficients, residual = csc.encode(sequence, nbNonzeroCoefs=8, minCoefficients=1e-6)
        c = coefficients.tocoo()
        self.assertTrue(coefficients.nnz == 6)
        self.assertTrue(np.array_equal(c.row, [32,48,64,96,128,192]))
        self.assertTrue(np.array_equal(c.col, [0,3,1,0,2,2]))
        self.assertTrue(np.allclose(c.data, [1.0,1.0,0.5,1.0,0.75,2.0]))
        self.assertTrue(np.allclose(residual, np.zeros_like(residual), atol=1e-6))
            
    def test_reconstruct(self):

        for tolerance in [0.5, 0.1, 0.001, 0.000001]:
        
            cmp = ConvolutionalMatchingPursuit()
            
            nbComponents = 32
            filterWidth = 9
            nbFeatures = 1
            D = normalize(np.random.random(size=(nbComponents, filterWidth, nbFeatures)), axis=(1,2))
            csc = ConvolutionalSparseCoder(D, approximator=cmp)
            
            sequence = np.random.random(size=(256,1))
            coefficients, residual = csc.encode(sequence, nbNonzeroCoefs=None, toleranceResidualScale=tolerance)
                
            sequenceRecons = csc.reconstruct(coefficients)
            residualRecons = sequence - sequenceRecons
            self.assertTrue(np.allclose(sequenceRecons.shape, sequence.shape))
            self.assertTrue(np.max(np.abs(residualRecons)) < tolerance)
            self.assertTrue(np.allclose(residual, residualRecons, atol=1e6))

class TestFunctions(unittest.TestCase):

    def test_extractRandomWindows(self):
        sequence = np.arange(100)
        windows = extractRandomWindows(sequence, nbWindows=16, width=10)
        self.assertTrue(np.array_equal(windows.shape, [16,10]))

    def test_extractWindows(self):
        sequences = np.arange(64).reshape((4,16))
        indices = np.array([0,1,4,7])
        windows = extractWindows(sequences, indices, width=5, centered=False)
        self.assertTrue(np.array_equal(windows.shape, [4,5]))
        self.assertTrue(np.array_equal(windows[0], [0,1,2,3,4]))
        self.assertTrue(np.array_equal(windows[1], [17,18,19,20,21]))
        self.assertTrue(np.array_equal(windows[2], [36,37,38,39,40]))
        self.assertTrue(np.array_equal(windows[3], [55,56,57,58,59]))
        
        windows = extractWindows(sequences, indices, width=6, centered=False)
        self.assertTrue(np.array_equal(windows.shape, [4,6]))
        self.assertTrue(np.array_equal(windows[0], [0,1,2,3,4,5]))
        self.assertTrue(np.array_equal(windows[1], [17,18,19,20,21,22]))
        self.assertTrue(np.array_equal(windows[2], [36,37,38,39,40,41]))
        self.assertTrue(np.array_equal(windows[3], [55,56,57,58,59,60]))
        
        indices = np.array([5,7,8,11])
        windows = extractWindows(sequences, indices, width=5, centered=True)
        self.assertTrue(np.array_equal(windows.shape, [4,5]))
        self.assertTrue(np.array_equal(windows[0], [3,4,5,6,7]))
        self.assertTrue(np.array_equal(windows[1], [21,22,23,24,25]))
        self.assertTrue(np.array_equal(windows[2], [38,39,40,41,42]))
        self.assertTrue(np.array_equal(windows[3], [57,58,59,60,61]))
        
        windows = extractWindows(sequences, indices, width=6, centered=True)
        self.assertTrue(np.array_equal(windows.shape, [4,6]))
        self.assertTrue(np.array_equal(windows[0], [3,4,5,6,7,8]))
        self.assertTrue(np.array_equal(windows[1], [21,22,23,24,25,26]))
        self.assertTrue(np.array_equal(windows[2], [38,39,40,41,42,43]))
        self.assertTrue(np.array_equal(windows[3], [57,58,59,60,61,62]))
        
    def test_convolve1d(self):
        
        sequence = np.arange(10)
        nbFilters = 4
        filterWidth = 5
        filters = np.random.uniform(size=(nbFilters, filterWidth))
        c = convolve1d(sequence, filters, padding='same')
        self.assertTrue(np.array_equal(c.shape, [len(sequence), nbFilters]))
        
        sequence = np.arange(10)
        nbFilters = 4
        filterWidth = 5
        filters = np.random.uniform(size=(nbFilters, filterWidth))
        c = convolve1d(sequence, filters, padding='valid')
        self.assertTrue(np.array_equal(c.shape, [len(sequence)-filterWidth+1, nbFilters]))
        
        filterWidth = 6
        filters = np.random.uniform(size=(nbFilters, filterWidth))
        c = convolve1d(sequence, filters, padding='same')
        self.assertTrue(np.array_equal(c.shape, [len(sequence), nbFilters]))
        
        filterWidth = 6
        filters = np.random.uniform(size=(nbFilters, filterWidth))
        c = convolve1d(sequence, filters, padding='valid')
        self.assertTrue(np.array_equal(c.shape, [len(sequence)-filterWidth+1, nbFilters]))

        nbFeatures = 2
        sequence = np.arange(16 * nbFeatures).reshape((16, nbFeatures))
        nbFilters = 4
        filterWidth = 5
        filters = np.random.uniform(size=(nbFilters, filterWidth, nbFeatures))
        c = convolve1d(sequence, filters, padding='same')
        self.assertTrue(np.array_equal(c.shape, [len(sequence), nbFilters]))
                 
        for nbFeatures in [1,2,5]:
            for filterWidth in [5,6]:
                sequence = np.arange(16 * nbFeatures).reshape((16, nbFeatures))
                filters = np.array([sequence[:filterWidth, :],
                                    sequence[1:filterWidth+1, :],
                                    sequence[2:filterWidth+2, :],
                                    sequence[3:filterWidth+3, :]])
                c = convolve1d(sequence, filters, padding='same')
                self.assertTrue(np.array_equal(c.shape, [sequence.shape[0], filters.shape[0]]))
                for i in range(filters.shape[0]):
                    if np.mod(filterWidth, 2) == 0:
                        self.assertTrue(np.allclose(c[i+filterWidth/2-1,i], np.sum(np.square(filters[i]))))
                    else:
                        self.assertTrue(np.allclose(c[i+filterWidth/2,i], np.sum(np.square(filters[i]))))
        
    def test_convolve1d_batch(self):
         
        sequences = np.tile(np.arange(10).reshape(1,10), (12,1))
        nbFilters = 4
        filterWidth = 5
        filters = np.random.uniform(size=(nbFilters, filterWidth))
        c = convolve1d_batch(sequences, filters, padding='same')
        self.assertTrue(np.array_equal(c.shape, [sequences.shape[0], sequences.shape[1], nbFilters]))
         
        sequences = np.tile(np.arange(10).reshape(1,10), (12,1))
        nbFilters = 4
        filterWidth = 5
        filters = np.random.uniform(size=(nbFilters, filterWidth))
        c = convolve1d_batch(sequences, filters, padding='valid')
        self.assertTrue(np.array_equal(c.shape, [sequences.shape[0], sequences.shape[1]-filterWidth+1, nbFilters]))
        
        sequences = np.tile(np.arange(10).reshape(1,10), (12,1))
        nbFilters = 4
        filterWidth = 6
        filters = np.random.uniform(size=(nbFilters, filterWidth))
        c = convolve1d_batch(sequences, filters, padding='same')
        self.assertTrue(np.array_equal(c.shape, [sequences.shape[0], sequences.shape[1], nbFilters]))
         
        sequences = np.tile(np.arange(10).reshape(1,10), (12,1))
        nbFilters = 4
        filterWidth = 6
        filters = np.random.uniform(size=(nbFilters, filterWidth))
        c = convolve1d_batch(sequences, filters, padding='valid')
        self.assertTrue(np.array_equal(c.shape, [sequences.shape[0], sequences.shape[1]-filterWidth+1, nbFilters]))
         
        for nbFeatures in [1,2,5]:
            for filterWidth in [5,6]:
                sequence = np.arange(16 * nbFeatures).reshape((16, nbFeatures))
                filters = np.array([sequence[:filterWidth, :],
                                    sequence[1:filterWidth+1, :],
                                    sequence[2:filterWidth+2, :],
                                    sequence[3:filterWidth+3, :]])
                sequences = np.tile(sequence.reshape(1,sequence.shape[0], sequence.shape[1]), (12,1,1))
                c = convolve1d_batch(sequences, filters, padding='same')
                self.assertTrue(np.array_equal(c.shape, [sequences.shape[0], sequences.shape[1], filters.shape[0]]))
                for i in range(filters.shape[0]):
                    for b in range(sequences.shape[0]):
                        if np.mod(filterWidth, 2) == 0:
                            self.assertTrue(np.allclose(c[b,i+filterWidth/2-1,i], np.sum(np.square(filters[i]))))
                        else:
                            self.assertTrue(np.allclose(c[b,i+filterWidth/2,i], np.sum(np.square(filters[i]))))
                
if __name__ == '__main__':
    unittest.main()
