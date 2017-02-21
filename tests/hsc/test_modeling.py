
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

from hsc.modeling import ConvolutionalMatchingPursuit, ConvolutionalSparseCoder, ConvolutionalDictionaryLearner, HierarchicalConvolutionalSparseCoder, HierarchicalConvolutionalMatchingPursuit, \
                         extractRandomWindows, convolve1d, convolve1d_batch, extractWindows, extractWindowsBatch, reconstructSignal, ConvolutionalNMF, HierarchicalConvolutionalSparseCoder, \
                         MptkConvolutionalMatchingPursuit, LoCOMP
from hsc.utils import normalize, overlapAdd
from hsc.dataset import MultilevelDictionary, MultilevelDictionaryGenerator, SignalGenerator

class TestConvolutionalNMF(unittest.TestCase):

    def test_computeCoefficients_1d(self):
        
        for filterWidth in [5,9,16]:
            cnmf = ConvolutionalNMF()
            sequence = np.random.random(size=(256,))
            D = normalize(np.random.random(size=(16,filterWidth)), axis=1)
            coefficients, residual = cnmf.computeCoefficients(sequence, D, nbMaxIterations=10)
            self.assertTrue(np.sum(np.square(residual)) < np.sum(np.square(sequence)))
        
    def test_computeCoefficients_2d(self):
        
        for filterWidth in [5,9,16]:
            nbFeatures = 7
            cnmf = ConvolutionalNMF()
            sequence = np.random.random(size=(64,nbFeatures))
            D = normalize(np.random.random(size=(16,15,nbFeatures)), axis=(1,2))
            coefficients, residual = cnmf.computeCoefficients(sequence, D, nbMaxIterations=10)
            self.assertTrue(np.sum(np.square(residual)) < np.sum(np.square(sequence)))
        
class TestConvolutionalDictionaryLearner(unittest.TestCase):

    def test_train_samples_1d(self):
        sequence = np.random.random(size=(256,))
        cdl = ConvolutionalDictionaryLearner(k=16, windowSize=5, algorithm='samples')
        D = cdl.train(sequence)
        self.assertTrue(np.array_equal(D.shape, [16,5]))

    def test_train_samples_2d(self):
        nbFeatures = 4
        sequence = np.random.random(size=(256,nbFeatures))
        cdl = ConvolutionalDictionaryLearner(k=16, windowSize=5, algorithm='samples')
        D = cdl.train(sequence)
        self.assertTrue(np.array_equal(D.shape, [16,5,nbFeatures]))

    def test_train_nmf_1d(self):
        sequence = np.random.random(size=(256,))
        cdl = ConvolutionalDictionaryLearner(k=16, windowSize=5, algorithm='nmf')
        D = cdl.train(sequence, nbMaxIterations=100, initMethod='random_samples')
        self.assertTrue(np.array_equal(D.shape, [16,5]))

    def test_train_nmf_2d(self):
        nbFeatures = 4
        sequence = np.random.random(size=(256,nbFeatures))
        cdl = ConvolutionalDictionaryLearner(k=16, windowSize=5, algorithm='nmf')
        D = cdl.train(sequence, nbMaxIterations=100, initMethod='random_samples')
        self.assertTrue(np.array_equal(D.shape, [16,5,nbFeatures]))

    def test_train_kmean_1d(self):
        
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

    def test_train_kmean_2d(self):
        nbFeatures = 4
        for initMethod in ['noise', 'random_samples']:
            sequence = np.random.random(size=(256,nbFeatures))
            cdl = ConvolutionalDictionaryLearner(k=16, windowSize=5, algorithm='kmean')
            D = cdl.train(sequence, nbRandomWindows=32, maxIterations=100, tolerance=0.0, initMethod=initMethod)
            self.assertTrue(np.array_equal(D.shape, [16,5,nbFeatures]))
            
        nbFeatures = 4
        for resetMethod in ['noise', 'random_samples', 'random_samples_average']:
            sequence = np.random.random(size=(256,nbFeatures))
            cdl = ConvolutionalDictionaryLearner(k=16, windowSize=5, algorithm='kmean')
            D = cdl.train(sequence, nbRandomWindows=32, maxIterations=100, tolerance=0.0, resetMethod=resetMethod)
            self.assertTrue(np.array_equal(D.shape, [16,5,nbFeatures]))

class TestMptkConvolutionalMatchingPursuit(unittest.TestCase):
    
    def test_computeCoefficients_1d(self):
        
        # 1D sequence
        nbNonzeroCoefs = 4
        for dsize in [1,2,3]:
            for dwidth in [3,5,6]:
                cmp = MptkConvolutionalMatchingPursuit(method='mp')
                sequence = np.random.random(size=(16,))
                D = normalize(np.random.random(size=(dsize,dwidth)), axis=1)
                coefficients, residual = cmp.computeCoefficients(sequence, D, nbNonzeroCoefs=nbNonzeroCoefs)
                self.assertTrue(coefficients.nnz == nbNonzeroCoefs)
                self.assertTrue(np.sum(np.square(residual)) < np.sum(np.square(sequence)))
        
    def test_computeCoefficients_2d(self):
        
        # 2D sequence
        nbNonzeroCoefs = 1
        nbFeatures = 7
        cmp = MptkConvolutionalMatchingPursuit(method='mp')
        sequence = np.random.random(size=(128,nbFeatures))
        D = normalize(np.random.random(size=(16,15,nbFeatures)), axis=(1,2))
        coefficients, residual = cmp.computeCoefficients(sequence, D, nbNonzeroCoefs=nbNonzeroCoefs)
        self.assertTrue(coefficients.nnz == nbNonzeroCoefs)
        self.assertTrue(np.sum(np.square(residual)) < np.sum(np.square(sequence)))
    
    def test_optimality_1d(self):
    
        N = 128     # Number of samples
        fs = 8000   # Sampling frequency
        f0 = 80
        f1 = 200
        sig1 = np.sin(np.arange(N).astype(np.float32)/fs*2*np.pi*f0) * np.hanning(N)
        sig2 = np.sin(np.arange(N).astype(np.float32)/fs*2*np.pi*f1) * np.hanning(N)
         
        # Normalize and create dictionary
        sig1 /= np.sqrt(np.sum(np.square(sig1)))
        sig2 /= np.sqrt(np.sum(np.square(sig2)))
        D = np.vstack([sig1, sig2])

        signal = np.zeros(8192)
        nnz = 32
        fIndices = np.random.randint(low=0, high=D.shape[0], size=(nnz,))
        positions = np.random.randint(low=0, high=len(signal) - D.shape[1], size=(nnz,))
        for fIdx, position in zip(fIndices, positions):
            signal[position:position+D.shape[1]] += D[fIdx,:]
        
        cmp = MptkConvolutionalMatchingPursuit(method='cmp')
        coefficients, residual = cmp.computeCoefficients(signal, D, toleranceSnr=40.0)
        self.assertTrue(coefficients.nnz < 2.0 * nnz)

class TestLoCOMP(unittest.TestCase):

    def test_computeCoefficients_1d(self):
        
        # 1D sequence
        nbNonzeroCoefs = 16
        cmp = LoCOMP()
        sequence = np.random.random(size=(64,))
        D = normalize(np.random.random(size=(16,15)), axis=1)
        coefficients, residual = cmp.computeCoefficients(sequence, D, nbNonzeroCoefs=nbNonzeroCoefs)
        self.assertTrue(coefficients.nnz == nbNonzeroCoefs)
        self.assertTrue(np.sum(np.square(residual)) < np.sum(np.square(sequence)))
        
        nbNonzeroCoefs = 4
        for dsize in [1,2,3,9]:
            for dwidth in [3,5,6]:
                cmp = LoCOMP()
                sequence = np.random.random(size=(16,))
                D = normalize(np.random.random(size=(dsize,dwidth)), axis=1)
                coefficients, residual = cmp.computeCoefficients(sequence, D, nbNonzeroCoefs=nbNonzeroCoefs)
                self.assertTrue(coefficients.nnz == nbNonzeroCoefs)
                self.assertTrue(np.sum(np.square(residual)) < np.sum(np.square(sequence)))
        
        # 1D sequence with blocking
        for nbBlocks in [1,2,8]:
            cmp = LoCOMP()
            sequence = np.random.random(size=(256,))
            D = normalize(np.random.random(size=(16,15)), axis=1)
            coefficients, residual = cmp.computeCoefficients(sequence, D, nbBlocks=nbBlocks)
            self.assertTrue(np.sum(np.square(residual)) < np.sum(np.square(sequence)))
        
    def test_computeCoefficients_2d(self):
        
        # 2D sequence
        nbNonzeroCoefs = 16
        nbFeatures = 7
        cmp = LoCOMP()
        sequence = np.random.random(size=(64,nbFeatures))
        D = normalize(np.random.random(size=(16,15,nbFeatures)), axis=(1,2))
        coefficients, residual = cmp.computeCoefficients(sequence, D, nbNonzeroCoefs=nbNonzeroCoefs)
        self.assertTrue(coefficients.nnz == nbNonzeroCoefs)
        self.assertTrue(np.sum(np.square(residual)) < np.sum(np.square(sequence)))

        for nbFeatures in [2,5,7,15]:
            cmp = LoCOMP()
            sequence = np.random.random(size=(64,nbFeatures))
            D = normalize(np.random.random(size=(16,15,nbFeatures)), axis=(1,2))
            coefficients, residual = cmp.computeCoefficients(sequence, D, nbNonzeroCoefs=nbNonzeroCoefs)
            self.assertTrue(np.sum(np.square(residual)) < np.sum(np.square(sequence)))

        # 2D sequence with blocking
        for nbBlocks in [1,2,8]:
            cmp = LoCOMP()
            sequence = np.random.random(size=(256,nbFeatures))
            D = normalize(np.random.random(size=(16,15,nbFeatures)), axis=(1,2))
            coefficients, residual = cmp.computeCoefficients(sequence, D, nbBlocks=nbBlocks)
            self.assertTrue(np.sum(np.square(residual)) < np.sum(np.square(sequence)))
   

class TestConvolutionalMatchingPursuit(unittest.TestCase):

    def test_computeCoefficients_1d(self):
        
        # 1D sequence
        nbNonzeroCoefs = 4
        for dsize in [1,2,3]:
            for dwidth in [3,5,6]:
                cmp = ConvolutionalMatchingPursuit()
                sequence = np.random.random(size=(16,))
                D = normalize(np.random.random(size=(dsize,dwidth)), axis=1)
                coefficients, residual = cmp.computeCoefficients(sequence, D, nbNonzeroCoefs=nbNonzeroCoefs)
                self.assertTrue(coefficients.nnz == nbNonzeroCoefs)
                self.assertTrue(np.sum(np.square(residual)) < np.sum(np.square(sequence)))
        
    def test_computeCoefficients_2d(self):
        
        # 2D sequence
        nbNonzeroCoefs = 16
        nbFeatures = 7
        cmp = ConvolutionalMatchingPursuit()
        sequence = np.random.random(size=(64,nbFeatures))
        D = normalize(np.random.random(size=(16,15,nbFeatures)), axis=(1,2))
        coefficients, residual = cmp.computeCoefficients(sequence, D, nbNonzeroCoefs=nbNonzeroCoefs)
        self.assertTrue(coefficients.nnz == nbNonzeroCoefs)
        self.assertTrue(np.sum(np.square(residual)) < np.sum(np.square(sequence)))

    def test_selectBestAtoms(self):
        
        # Odd filter size, with and without offset, fixed number of blocks (no interference expected)
        filterWidth = 5
        nbBlocks = 4
        cmp = ConvolutionalMatchingPursuit()
        innerProducts = np.arange(256).reshape((64,4)).astype(np.float)
        innerProducts[-1] = innerProducts[-1][::-1]
        atoms = cmp._selectBestAtoms(innerProducts, filterWidth, nbBlocks, offset=False)
        t = [atom.position for atom in atoms]
        fIdx = [atom.index for atom in atoms]
        self.assertTrue(len(t) == nbBlocks)
        self.assertTrue(len(fIdx) == nbBlocks)
        self.assertTrue(np.array_equal(t,[63,47,31,15]))
        self.assertTrue(np.array_equal(fIdx,[0,3,3,3]))
        self.assertTrue(np.allclose(np.abs(innerProducts[t[0], fIdx[0]]), np.abs(np.max(innerProducts))))
        atoms = cmp._selectBestAtoms(innerProducts, filterWidth, nbBlocks, offset=True)
        t = [atom.position for atom in atoms]
        fIdx = [atom.index for atom in atoms]
        self.assertTrue(len(t) == nbBlocks+1)
        self.assertTrue(len(fIdx) == nbBlocks+1)
        self.assertTrue(np.array_equal(t,[63,55,39,23,7]))
        self.assertTrue(np.array_equal(fIdx,[0,3,3,3,3]))
        self.assertTrue(np.allclose(np.abs(innerProducts[t[0], fIdx[0]]), np.abs(np.max(innerProducts))))
        
        # Odd filter size, no offset, automatic number of blocks (no interference expected)
        filterWidth = 3
        nbBlocks = None
        cmp = ConvolutionalMatchingPursuit()
        innerProducts = np.arange(256).reshape((64,4)).astype(np.float)
        innerProducts[-1] = innerProducts[-1][::-1]
        atoms = cmp._selectBestAtoms(innerProducts, filterWidth, nbBlocks, offset=False)
        t = [atom.position for atom in atoms]
        fIdx = [atom.index for atom in atoms]
        self.assertTrue(len(t) == 6)
        self.assertTrue(len(fIdx) == 6)
        self.assertTrue(np.array_equal(t,[63,59,47,35,23,11]))
        self.assertTrue(np.array_equal(fIdx,[0,3,3,3,3,3]))
        self.assertTrue(np.allclose(np.abs(innerProducts[t[0], fIdx[0]]), np.abs(np.max(innerProducts))))
        
        # Odd filter size, no offset, fixed number of blocks (interference expected)
        filterWidth = 5
        nbBlocks = 5
        cmp = ConvolutionalMatchingPursuit()
        innerProducts = np.arange(256).reshape((64,4)).astype(np.float)
        innerProducts[-1] = innerProducts[-1][::-1]
        atoms = cmp._selectBestAtoms(innerProducts, filterWidth, nbBlocks, offset=False)
        t = [atom.position for atom in atoms]
        fIdx = [atom.index for atom in atoms]
        self.assertTrue(len(t) == nbBlocks)
        self.assertTrue(len(fIdx) == nbBlocks)
        self.assertTrue(np.array_equal(t,[59,47,35,23,11])) # Missing 63 because of interference
        self.assertTrue(np.array_equal(fIdx,[3,3,3,3,3]))
        self.assertTrue(not np.allclose(np.abs(innerProducts[t[0], fIdx[0]]), np.abs(np.max(innerProducts))))
        
class TestConvolutionalSparseCoder(unittest.TestCase):

    def test_encode_1d_scale_tolerance(self):
        
        # 1D sequence, variable scale tolerance on residual
        for tolerance in [0.5, 0.1, 0.001, 0.000001]:
            cmp = ConvolutionalMatchingPursuit()
            nbComponents = 32
            filterWidth = 9
            D = normalize(np.random.random(size=(nbComponents, filterWidth)), axis=1)
            csc = ConvolutionalSparseCoder(D, approximator=cmp)
            
            sequence = np.random.random(size=(128,))
            coefficients, residual = csc.encode(sequence, nbNonzeroCoefs=None, toleranceResidualScale=tolerance)
            self.assertTrue(scipy.sparse.issparse(coefficients))
            self.assertTrue(coefficients.nnz > 0)
            self.assertTrue(np.max(np.abs(residual)) < tolerance)

    def test_encode_1d_snr_tolerance(self):
            
        # 1D sequence, variable SNR tolerance on residual
        for tolerance in [5, 10, 20, 50]:
            cmp = ConvolutionalMatchingPursuit()
            nbComponents = 32
            filterWidth = 9
            D = normalize(np.random.random(size=(nbComponents, filterWidth)), axis=1)
            csc = ConvolutionalSparseCoder(D, approximator=cmp)
            
            sequence = np.random.random(size=(128,))
            coefficients, residual = csc.encode(sequence, nbNonzeroCoefs=None, toleranceSnr=tolerance)
            snr = 10.0*np.log10(np.sum(np.square(sequence))/np.sum(np.square(residual)))
            self.assertTrue(scipy.sparse.issparse(coefficients))
            self.assertTrue(coefficients.nnz > 0)
            self.assertTrue(snr >= tolerance)
    
    def test_encode_2d_features(self):
    
        # 2D sequence, variable number of features
        tolerance = 0.01
        for nbFeatures in [1, 4, 8, 11]:
            cmp = ConvolutionalMatchingPursuit()
            nbComponents = 32
            filterWidth = 9
            D = normalize(np.random.random(size=(nbComponents, filterWidth, nbFeatures)), axis=(1,2))
            csc = ConvolutionalSparseCoder(D, approximator=cmp)
            
            sequence = np.random.random(size=(128, nbFeatures))
            coefficients, residual = csc.encode(sequence, nbNonzeroCoefs=None, toleranceResidualScale=tolerance)
            self.assertTrue(scipy.sparse.issparse(coefficients))
            self.assertTrue(coefficients.nnz > 0)
            self.assertTrue(np.max(np.abs(residual)) < tolerance)
    
    def test_encode_1d_optimality_cmp(self):
        
        # Toy problem to solve
        cmp = ConvolutionalMatchingPursuit()
        nbComponents = 4
        filterWidth = 32
        D = normalize(np.random.random(size=(nbComponents, filterWidth)), axis=1)
        csc = ConvolutionalSparseCoder(D, approximator=cmp)
        
        coefficientsRef = scipy.sparse.coo_matrix(([1.0,1.0,0.5,1.0,0.75,2.0], 
                                                ([32,48,64,96,128,192], [0,3,1,0,2,2])),
                                               shape = (256,nbComponents))
        sequence = reconstructSignal(coefficientsRef, D)
        
        coefficients, residual = csc.encode(sequence, nbNonzeroCoefs=8, minCoefficients=1e-6)
        self.assertTrue(coefficients.nnz == coefficientsRef.nnz)
        self.assertTrue(np.allclose(coefficients.toarray(), coefficientsRef.toarray()))
        self.assertTrue(np.allclose(residual, np.zeros_like(residual), atol=1e-6))
            
    def test_encode_1d_optimality_locomp(self):
        
        # Toy problem to solve
        cmp = LoCOMP()
        nbComponents = 4
        filterWidth = 32
        D = normalize(np.random.random(size=(nbComponents, filterWidth)), axis=1)
        csc = ConvolutionalSparseCoder(D, approximator=cmp)
        
        coefficientsRef = scipy.sparse.coo_matrix(([1.0,1.0,0.5,1.0,0.75,2.0], 
                                                ([32,48,64,96,128,192], [0,3,1,0,2,2])),
                                               shape = (256,nbComponents))
        sequence = reconstructSignal(coefficientsRef, D)
        
        coefficients, residual = csc.encode(sequence, minCoefficients=1e-10)
        self.assertTrue(coefficients.nnz == coefficientsRef.nnz)
        self.assertTrue(np.allclose(coefficients.toarray(), coefficientsRef.toarray(), atol=1e-1))
        self.assertTrue(np.allclose(residual, np.zeros_like(residual), atol=1e-6))
            
    def test_encode_2d_optimality_locomp(self):
        
        # Toy problem to solve
        cmp = LoCOMP()
        nbComponents = 4
        filterWidth = 32
        nbFeatures = 7
        D = normalize(np.random.random(size=(nbComponents, filterWidth, nbFeatures)), axis=(1,2))
        csc = ConvolutionalSparseCoder(D, approximator=cmp)
        
        coefficientsRef = scipy.sparse.coo_matrix(([1.0,1.0,0.5,1.0,0.75,2.0], 
                                                ([32,48,64,96,128,192], [0,3,1,0,2,2])),
                                               shape = (256,nbComponents))
        sequence = reconstructSignal(coefficientsRef, D)
        
        coefficients, residual = csc.encode(sequence, minCoefficients=1e-10)
        self.assertTrue(coefficients.nnz == coefficientsRef.nnz)
        self.assertTrue(np.allclose(coefficients.toarray(), coefficientsRef.toarray(), atol=1e-1))
        self.assertTrue(np.allclose(residual, np.zeros_like(residual), atol=1e-6))
            
    def test_reconstruct_1d(self):
        
        # 1D sequence, variable tolerance for residual
        for tolerance in [0.5, 0.1, 0.001]:
            for filterWidth in [7, 10]:
                cmp = ConvolutionalMatchingPursuit()
                nbComponents = 32
                D = normalize(np.random.random(size=(nbComponents, filterWidth)), axis=1)
                csc = ConvolutionalSparseCoder(D, approximator=cmp)
                
                sequence = np.random.random(size=(256,))
                coefficients, residual = csc.encode(sequence, nbNonzeroCoefs=None, toleranceResidualScale=tolerance)
                    
                # Using sparse format
                sequenceRecons = csc.reconstruct(coefficients)
                self.assertTrue(np.allclose(sequenceRecons.shape, sequence.shape))
                residualRecons = sequence - sequenceRecons
                self.assertTrue(np.max(np.abs(residualRecons)) < tolerance)
                self.assertTrue(np.allclose(residual, residualRecons, atol=1e6))
                
                # Using dense format
                sequenceRecons = csc.reconstruct(coefficients.toarray())
                self.assertTrue(np.allclose(sequenceRecons.shape, sequence.shape))
                residualRecons = sequence - sequenceRecons
                self.assertTrue(np.max(np.abs(residualRecons)) < tolerance)
                self.assertTrue(np.allclose(residual, residualRecons, atol=1e6))

    def test_reconstruct_2d(self):

        # 2D sequence, variable tolerance for residual
        for tolerance in [0.5, 0.1, 0.001]:
            for filterWidth in [7, 10]:
                cmp = ConvolutionalMatchingPursuit()
                nbComponents = 32
                nbFeatures = 7
                D = normalize(np.random.random(size=(nbComponents, filterWidth, nbFeatures)), axis=(1,2))
                csc = ConvolutionalSparseCoder(D, approximator=cmp)
                
                sequence = np.random.random(size=(256, nbFeatures))
                coefficients, residual = csc.encode(sequence, nbNonzeroCoefs=None, toleranceResidualScale=tolerance)
                    
                # Using sparse format
                sequenceRecons = csc.reconstruct(coefficients)
                residualRecons = sequence - sequenceRecons
                self.assertTrue(np.allclose(sequenceRecons.shape, sequence.shape))
                self.assertTrue(np.max(np.abs(residualRecons)) < tolerance)
                self.assertTrue(np.allclose(residual, residualRecons, atol=1e6))
    
                # Using dense format
                sequenceRecons = csc.reconstruct(coefficients.toarray())
                self.assertTrue(np.allclose(sequenceRecons.shape, sequence.shape))
                residualRecons = sequence - sequenceRecons
                self.assertTrue(np.max(np.abs(residualRecons)) < tolerance)
                self.assertTrue(np.allclose(residual, residualRecons, atol=1e6))

class TestFunctions(unittest.TestCase):

    def test_extractRandomWindows_1d(self):
        sequence = np.arange(100)
        windows = extractRandomWindows(sequence, nbWindows=16, width=10)
        self.assertTrue(np.array_equal(windows.shape, [16,10]))
 
    def test_extractRandomWindows_2d(self):
        nbFeatures = 2
        sequence = np.arange(50*nbFeatures).reshape((50,nbFeatures))
        windows = extractRandomWindows(sequence, nbWindows=16, width=10)
        self.assertTrue(np.array_equal(windows.shape, [16,10,nbFeatures]))

    def test_extractWindows_1d(self):

        # Odd width, not centered
        sequence = np.arange(16)
        indices = np.array([0,1,4,7])
        windows = extractWindows(sequence, indices, width=5, centered=False)
        self.assertTrue(np.array_equal(windows.shape, [4,5]))
        self.assertTrue(np.array_equal(windows[0], [0,1,2,3,4]))
        self.assertTrue(np.array_equal(windows[1], [1,2,3,4,5]))
        self.assertTrue(np.array_equal(windows[2], [4,5,6,7,8]))
        self.assertTrue(np.array_equal(windows[3], [7,8,9,10,11]))
        
        # Even width, not centered
        sequence = np.arange(16)
        indices = np.array([0,1,4,7])
        windows = extractWindows(sequence, indices, width=6, centered=False)
        self.assertTrue(np.array_equal(windows.shape, [4,6]))
        self.assertTrue(np.array_equal(windows[0], [0,1,2,3,4,5]))
        self.assertTrue(np.array_equal(windows[1], [1,2,3,4,5,6]))
        self.assertTrue(np.array_equal(windows[2], [4,5,6,7,8,9]))
        self.assertTrue(np.array_equal(windows[3], [7,8,9,10,11,12]))
        
        # Odd width, centered
        sequence = np.arange(16)
        indices = np.array([2,3,6,9])
        windows = extractWindows(sequence, indices, width=5, centered=True)
        self.assertTrue(np.array_equal(windows.shape, [4,5]))
        self.assertTrue(np.array_equal(windows[0], [0,1,2,3,4]))
        self.assertTrue(np.array_equal(windows[1], [1,2,3,4,5]))
        self.assertTrue(np.array_equal(windows[2], [4,5,6,7,8]))
        self.assertTrue(np.array_equal(windows[3], [7,8,9,10,11]))
        
        # Even width, centered
        sequence = np.arange(16)
        indices = np.array([2,3,6,9])
        windows = extractWindows(sequence, indices, width=6, centered=True)
        self.assertTrue(np.array_equal(windows.shape, [4,6]))
        self.assertTrue(np.array_equal(windows[0], [0,1,2,3,4,5]))
        self.assertTrue(np.array_equal(windows[1], [1,2,3,4,5,6]))
        self.assertTrue(np.array_equal(windows[2], [4,5,6,7,8,9]))
        self.assertTrue(np.array_equal(windows[3], [7,8,9,10,11,12]))

    def test_extractWindows_2d(self):
        
        # Odd width, not centered
        nbFeatures = 2
        sequence = np.arange(16*nbFeatures).reshape((16,nbFeatures))
        indices = np.array([0,1,4,7])
        windows = extractWindows(sequence, indices, width=5, centered=False)
        self.assertTrue(np.array_equal(windows.shape, [4,5,nbFeatures]))
        self.assertTrue(np.array_equal(windows[0], np.array([[0,1],[2,3],[4,5],[6,7],[8,9]])))
        self.assertTrue(np.array_equal(windows[1], np.array([[2,3],[4,5],[6,7],[8,9],[10,11]])))
        self.assertTrue(np.array_equal(windows[2], np.array([[8,9],[10,11],[12,13],[14,15],[16,17]])))
        self.assertTrue(np.array_equal(windows[3], np.array([[14,15],[16,17],[18,19],[20,21],[22,23]])))
        
        # Even width, not centered
        nbFeatures = 2
        sequence = np.arange(16*nbFeatures).reshape((16,nbFeatures))
        indices = np.array([0,1,4,7])
        windows = extractWindows(sequence, indices, width=6, centered=False)
        self.assertTrue(np.array_equal(windows.shape, [4,6,nbFeatures]))
        self.assertTrue(np.array_equal(windows[0], np.array([[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]])))
        self.assertTrue(np.array_equal(windows[1], np.array([[2,3],[4,5],[6,7],[8,9],[10,11],[12,13]])))
        self.assertTrue(np.array_equal(windows[2], np.array([[8,9],[10,11],[12,13],[14,15],[16,17],[18,19]])))
        self.assertTrue(np.array_equal(windows[3], np.array([[14,15],[16,17],[18,19],[20,21],[22,23],[24,25]])))
        
        # Odd width, centered
        nbFeatures = 2
        sequence = np.arange(16*nbFeatures).reshape((16,nbFeatures))
        indices = np.array([2,3,6,9])
        windows = extractWindows(sequence, indices, width=5, centered=True)
        self.assertTrue(np.array_equal(windows.shape, [4,5,nbFeatures]))
        self.assertTrue(np.array_equal(windows[0], np.array([[0,1],[2,3],[4,5],[6,7],[8,9]])))
        self.assertTrue(np.array_equal(windows[1], np.array([[2,3],[4,5],[6,7],[8,9],[10,11]])))
        self.assertTrue(np.array_equal(windows[2], np.array([[8,9],[10,11],[12,13],[14,15],[16,17]])))
        self.assertTrue(np.array_equal(windows[3], np.array([[14,15],[16,17],[18,19],[20,21],[22,23]])))
        
        # Even width, centered
        nbFeatures = 2
        sequence = np.arange(16*nbFeatures).reshape((16,nbFeatures))
        indices = np.array([2,3,6,9])
        windows = extractWindows(sequence, indices, width=6, centered=True)
        self.assertTrue(np.array_equal(windows.shape, [4,6,nbFeatures]))
        self.assertTrue(np.array_equal(windows[0], np.array([[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]])))
        self.assertTrue(np.array_equal(windows[1], np.array([[2,3],[4,5],[6,7],[8,9],[10,11],[12,13]])))
        self.assertTrue(np.array_equal(windows[2], np.array([[8,9],[10,11],[12,13],[14,15],[16,17],[18,19]])))
        self.assertTrue(np.array_equal(windows[3], np.array([[14,15],[16,17],[18,19],[20,21],[22,23],[24,25]])))

    def test_extractWindowsBatch_1d(self):
        
        # Odd width, not centered
        sequences = np.arange(64).reshape((4,16))
        indices = np.array([0,1,4,7])
        windows = extractWindowsBatch(sequences, indices, width=5, centered=False)
        self.assertTrue(np.array_equal(windows.shape, [4,5]))
        self.assertTrue(np.array_equal(windows[0], [0,1,2,3,4]))
        self.assertTrue(np.array_equal(windows[1], [17,18,19,20,21]))
        self.assertTrue(np.array_equal(windows[2], [36,37,38,39,40]))
        self.assertTrue(np.array_equal(windows[3], [55,56,57,58,59]))
        
        # Even width, not centered
        sequences = np.arange(64).reshape((4,16))
        indices = np.array([0,1,4,7])
        windows = extractWindowsBatch(sequences, indices, width=6, centered=False)
        self.assertTrue(np.array_equal(windows.shape, [4,6]))
        self.assertTrue(np.array_equal(windows[0], [0,1,2,3,4,5]))
        self.assertTrue(np.array_equal(windows[1], [17,18,19,20,21,22]))
        self.assertTrue(np.array_equal(windows[2], [36,37,38,39,40,41]))
        self.assertTrue(np.array_equal(windows[3], [55,56,57,58,59,60]))
        
        # Odd width, centered
        indices = np.array([5,7,8,11])
        windows = extractWindowsBatch(sequences, indices, width=5, centered=True)
        self.assertTrue(np.array_equal(windows.shape, [4,5]))
        self.assertTrue(np.array_equal(windows[0], [3,4,5,6,7]))
        self.assertTrue(np.array_equal(windows[1], [21,22,23,24,25]))
        self.assertTrue(np.array_equal(windows[2], [38,39,40,41,42]))
        self.assertTrue(np.array_equal(windows[3], [57,58,59,60,61]))
        
        # Even width, centered
        windows = extractWindowsBatch(sequences, indices, width=6, centered=True)
        self.assertTrue(np.array_equal(windows.shape, [4,6]))
        self.assertTrue(np.array_equal(windows[0], [3,4,5,6,7,8]))
        self.assertTrue(np.array_equal(windows[1], [21,22,23,24,25,26]))
        self.assertTrue(np.array_equal(windows[2], [38,39,40,41,42,43]))
        self.assertTrue(np.array_equal(windows[3], [57,58,59,60,61,62]))
        
    def test_extractWindowsBatch_2d(self):
        
        # Odd width, not centered
        nbFeatures = 2
        sequences = np.arange(64*nbFeatures).reshape((4,16,nbFeatures))
        indices = np.array([0,1,4,7])
        windows = extractWindowsBatch(sequences, indices, width=5, centered=False)
        self.assertTrue(np.array_equal(windows.shape, [4,5,nbFeatures]))
        self.assertTrue(np.array_equal(windows[0], np.array([[0,1],[2,3],[4,5],[6,7],[8,9]])))
        self.assertTrue(np.array_equal(windows[1], 32 + np.array([[2,3],[4,5],[6,7],[8,9],[10,11]])))
        self.assertTrue(np.array_equal(windows[2], 64 + np.array([[8,9],[10,11],[12,13],[14,15],[16,17]])))
        self.assertTrue(np.array_equal(windows[3], 96 + np.array([[14,15],[16,17],[18,19],[20,21],[22,23]])))
        
        # Even width, not centered
        nbFeatures = 2
        sequences = np.arange(64*nbFeatures).reshape((4,16,nbFeatures))
        indices = np.array([0,1,4,7])
        windows = extractWindowsBatch(sequences, indices, width=6, centered=False)
        self.assertTrue(np.array_equal(windows.shape, [4,6,nbFeatures]))
        self.assertTrue(np.array_equal(windows[0], np.array([[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]])))
        self.assertTrue(np.array_equal(windows[1], 32 + np.array([[2,3],[4,5],[6,7],[8,9],[10,11],[12,13]])))
        self.assertTrue(np.array_equal(windows[2], 64 + np.array([[8,9],[10,11],[12,13],[14,15],[16,17],[18,19]])))
        self.assertTrue(np.array_equal(windows[3], 96 + np.array([[14,15],[16,17],[18,19],[20,21],[22,23],[24,25]])))
        
        # Odd width, centered
        nbFeatures = 2
        sequences = np.arange(64*nbFeatures).reshape((4,16,nbFeatures))
        indices = np.array([2,3,6,9])
        windows = extractWindowsBatch(sequences, indices, width=5, centered=True)
        self.assertTrue(np.array_equal(windows.shape, [4,5,nbFeatures]))
        self.assertTrue(np.array_equal(windows[0], np.array([[0,1],[2,3],[4,5],[6,7],[8,9]])))
        self.assertTrue(np.array_equal(windows[1], 32 + np.array([[2,3],[4,5],[6,7],[8,9],[10,11]])))
        self.assertTrue(np.array_equal(windows[2], 64 + np.array([[8,9],[10,11],[12,13],[14,15],[16,17]])))
        self.assertTrue(np.array_equal(windows[3], 96 + np.array([[14,15],[16,17],[18,19],[20,21],[22,23]])))
        
        # Even width, centered
        nbFeatures = 2
        sequences = np.arange(64*nbFeatures).reshape((4,16,nbFeatures))
        indices = np.array([2,3,6,9])
        windows = extractWindowsBatch(sequences, indices, width=6, centered=True)
        self.assertTrue(np.array_equal(windows.shape, [4,6,nbFeatures]))
        self.assertTrue(np.array_equal(windows[0], np.array([[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]])))
        self.assertTrue(np.array_equal(windows[1], 32 + np.array([[2,3],[4,5],[6,7],[8,9],[10,11],[12,13]])))
        self.assertTrue(np.array_equal(windows[2], 64 + np.array([[8,9],[10,11],[12,13],[14,15],[16,17],[18,19]])))
        self.assertTrue(np.array_equal(windows[3], 96 + np.array([[14,15],[16,17],[18,19],[20,21],[22,23],[24,25]])))
        
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
                
    def test_reconstructSignal_1d(self):
        
        nbComponents = 4
        filterWidth = 32
        D = normalize(np.random.random(size=(nbComponents, filterWidth)), axis=1)
        
        rows = [32,48,64,96,128,192]
        cols = [0,3,1,0,2,2]
        data = [1.0,1.0,0.5,1.0,0.75,2.0]
        sequence = np.zeros((256,), dtype=np.float32)
        for i,j,c in zip(rows, cols, data):
            overlapAdd(sequence, c*D[j], t=i, copy=False)
        coefficients = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(sequence.shape[0], nbComponents))
                                     
        # Using sparse format                       
        reconstruction = reconstructSignal(coefficients, D)
        self.assertTrue(np.array_equal(reconstruction.shape, sequence.shape))
        self.assertTrue(np.allclose(reconstruction, sequence))
        
        # Using dense format                       
        reconstruction = reconstructSignal(coefficients.toarray(), D)
        self.assertTrue(np.array_equal(reconstruction.shape, sequence.shape))
        self.assertTrue(np.allclose(reconstruction, sequence))
                
    def test_reconstructSignal_2d(self):
        
        nbComponents = 4
        nbFeatures = 2
        filterWidth = 32
        D = normalize(np.random.random(size=(nbComponents, filterWidth, nbFeatures)), axis=1)
        
        rows = [32,48,64,96,128,192]
        cols = [0,3,1,0,2,2]
        data = [1.0,1.0,0.5,1.0,0.75,2.0]
        sequence = np.zeros((256, nbFeatures), dtype=np.float32)
        for i,j,c in zip(rows, cols, data):
            overlapAdd(sequence, c*D[j], t=i, copy=False)
        coefficients = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(sequence.shape[0], nbComponents))
                                     
        # Using sparse format
        reconstruction = reconstructSignal(coefficients, D)
        self.assertTrue(np.array_equal(reconstruction.shape, sequence.shape))
        self.assertTrue(np.allclose(reconstruction, sequence))
        
        # Using dense format                       
        reconstruction = reconstructSignal(coefficients.toarray(), D)
        self.assertTrue(np.array_equal(reconstruction.shape, sequence.shape))
        self.assertTrue(np.allclose(reconstruction, sequence))
                
class TestHierarchicalConvolutionalMatchingPursuit(unittest.TestCase):
           
    def test_computeCoefficients(self):
        
        mldg = MultilevelDictionaryGenerator()
        multilevelDict = mldg.generate(scales=[16,32,64], counts=[16,24,48], decompositionSize=4, multilevelDecomposition=False, maxNbPatternsConsecutiveRejected=10)
        
        nbSamples = 1024
        rates = [1e-3, 1e-3, 1e-3]
        generator = SignalGenerator(multilevelDict, rates)
        events, rates = generator.generateEvents(nbSamples, minimumCompressionRatio=0.50)
        signal = generator.generateSignalFromEvents(events, nbSamples)
        
        # NOTE: methods 'mptk-cmp' and 'mptk-mp' do not converge quickly for this test 
        for method in ['cmp', 'locomp']:
            
            hcmp = HierarchicalConvolutionalMatchingPursuit(method)
            coefficients, residual = hcmp.computeCoefficients(signal, multilevelDict.withSingletonBases(), toleranceSnr=[20,10,10], nbBlocks=1, alpha=0.1, singletonWeight=0.9)
            self.assertTrue(len(coefficients) == 3)
            self.assertTrue(np.array_equal(residual.shape, signal.shape))
            self.assertTrue(np.max(np.abs(residual)) < np.max(np.abs(signal)))
 
class TestHierarchicalConvolutionalSparseCoder(unittest.TestCase):
 
    def test_encode_1d(self):
        
        mldg = MultilevelDictionaryGenerator()
        multilevelDict = mldg.generate(scales=[16,32,64], counts=[16,24,48], decompositionSize=4, multilevelDecomposition=False, maxNbPatternsConsecutiveRejected=10)
        
        nbSamples = 1024
        rates = [1e-3, 1e-3, 1e-3]
        generator = SignalGenerator(multilevelDict, rates)
        events, rates = generator.generateEvents(nbSamples, minimumCompressionRatio=0.50)
        signal = generator.generateSignalFromEvents(events, nbSamples)
        
        # NOTE: methods 'mptk-cmp' and 'mptk-mp' do not converge quickly for this test 
        for method in ['cmp', 'locomp']:
        
            hcmp = HierarchicalConvolutionalMatchingPursuit(method)
            hsc = HierarchicalConvolutionalSparseCoder(multilevelDict, hcmp)
            coefficients, residual = hsc.encode(signal, toleranceSnr=[20,10,10], nbBlocks=1, alpha=0.1, singletonWeight=0.9)
            self.assertTrue(len(coefficients) == 3)
            self.assertTrue(np.array_equal(residual.shape, signal.shape))
            self.assertTrue(np.max(np.abs(residual)) < np.max(np.abs(signal)))
            
            reconstruction = hsc.reconstruct(coefficients)
            self.assertTrue(np.array_equal(signal.shape, reconstruction.shape))
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    np.seterr(all='raise')
    unittest.main()
