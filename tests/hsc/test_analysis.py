
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
import numpy as np
import scipy
import scipy.sparse

from hsc.dataset import MultilevelDictionary, MultilevelDictionaryGenerator
from hsc.analysis import calculateBitForLevels, calculateInformationRate, calculateMultilevelInformationRates, calculateBitForDatatype, visualizeDistributionRatios, calculateDistributionRatios, visualizeInformationRates, visualizeEnergies, visualizeInformationRatesOptimality

class TestFunctions(unittest.TestCase):

    def test_calculateBitForDatatype(self):
        c_bits = calculateBitForDatatype(dtype=np.float32)
        self.assertTrue(c_bits == 32)
        c_bits = calculateBitForDatatype(dtype=np.float64)
        self.assertTrue(c_bits == 64)
        c_bits = calculateBitForDatatype(dtype=np.int32)
        self.assertTrue(c_bits == 32)
        c_bits = calculateBitForDatatype(dtype=np.int16)
        self.assertTrue(c_bits == 16)
        
    def test_calculateBitForLevels(self):
        mldg = MultilevelDictionaryGenerator()
        multilevelDict = mldg.generate(scales=[32,64], counts=[8, 16])
        bits = calculateBitForLevels(multilevelDict, dtype=np.float32)
        self.assertTrue(len(bits) == multilevelDict.getNbLevels())
        # First level: 1bit scale + 3 bit index + 32bit value = 36 bits
        # Second level: 1bit scale + 4 bit index + 32bit value = 37 bits
        self.assertTrue(np.allclose(bits, [36, 37]))
        
        bits = calculateBitForLevels(multilevelDict, dtype=np.int16)
        self.assertTrue(len(bits) == multilevelDict.getNbLevels())
        # First level: 1bit scale + 3 bit index + 16bit value = 20 bits
        # Second level: 1bit scale + 4 bit index + 16bit value = 21 bits
        self.assertTrue(np.allclose(bits, [20, 21]))
        
    def test_calculateInformationRate(self):
        rates = [0.01]
        mldg = MultilevelDictionaryGenerator()
        multilevelDict = mldg.generate(scales=[32], counts=[8])
        avgInfoRate = calculateInformationRate(multilevelDict, rates)
        self.assertTrue(np.isscalar(avgInfoRate))
        # First level: 0bit scale + 3 bit index + 32bit value = 35 bits
        self.assertTrue(np.allclose(avgInfoRate, 35 * 0.01))
        
        rates = [0.01, 0.02]
        multilevelDict = mldg.generate(scales=[32,64], counts=[8, 16])
        avgInfoRate = calculateInformationRate(multilevelDict, rates)
        self.assertTrue(np.isscalar(avgInfoRate))
        # First level: 1bit scale + 3 bit index + 32bit value = 36 bits
        # Second level: 1bit scale + 4 bit index + 32bit value = 37 bits
        self.assertTrue(np.allclose(avgInfoRate, 36 * 0.01 + 37 * 0.02))
        
    def test_calculateMultilevelInformationRates(self):
        
        rates = [0.01, 0.02]
        mldg = MultilevelDictionaryGenerator()
        multilevelDict = mldg.generate(scales=[32,64], counts=[8, 16], decompositionSize=3)
        avgInfoRates = calculateMultilevelInformationRates(multilevelDict, rates, dtype=np.float32)
        # First level: 1bit scale + 3 bit index + 32bit value = 36 bits
        # Second level: 1bit scale + 4 bit index + 32bit value = 37 bits
        self.assertTrue(np.allclose(avgInfoRates, [36 * 0.01 * 8 + 36 * 0.02 * 3 * 16, 
                                                   36 * 0.01 * 8 + 37 * 0.02 * 16]))
        
        rates = [0.0001, 0.0002, 0.0004, 0.0004]
        multilevelDict = mldg.generate(scales=[32,64,128,256], counts=[8, 16, 32, 64], decompositionSize=4)
        avgInfoRates = calculateMultilevelInformationRates(multilevelDict, rates, dtype=np.float32)
        self.assertTrue(np.array_equal(np.argsort(avgInfoRates), np.arange(len(avgInfoRates))[::-1]))
        
    def test_calculateDistributionRatios(self):
        
        indices = [10,20,30,40,50,90]
        filters = [0,1,1,5,6,7]
        l0 = scipy.sparse.lil_matrix((100,8))
        l0[indices, filters] = np.random.random(len(indices))
        
        indices = [15,25,30,75]
        filters = [1,5,8,15]
        l1 = scipy.sparse.lil_matrix((100,16))
        l1[indices, filters] = np.random.random(len(indices))
        
        coefficients = [l0.tocsr(), l1.tocsr()]
        distribution = calculateDistributionRatios(coefficients)
        self.assertTrue(np.allclose(distribution, [0.6, 0.4]))
        
    def test_visualizeDistributionRatios(self):
        weights = np.array([0.1, 0.5, 0.75, 1.0, 5.0])
        distributions = np.array([[0.8,0.1,0.1],
                                  [0.8,0.15,0.05],
                                  [0.7,0.2,0.1],
                                  [0.6,0.3,0.1],
                                  [0.5,0.3,0.2]])
        fig = visualizeDistributionRatios(weights, distributions)
        self.assertTrue(fig is not None)
    
    def test_visualizeInformationRates(self):
        weights = np.array([0.1, 0.5, 0.75, 1.0, 5.0])
        sparseInfoRates = np.array([15.0, 12.5, 8.3, 6.4, 4.2])
        fig = visualizeInformationRates(weights, sparseInfoRates)
        self.assertTrue(fig is not None)
        
        fig = visualizeInformationRates(weights, sparseInfoRates, showAsBars=True)
        self.assertTrue(fig is not None)
    
    def test_visualizeEnergies(self):
        weights = np.array([0.1, 0.5, 0.75, 1.0, 5.0])
        energies = np.array([15.0, 12.5, 8.3, 6.4, 4.2])
        signalEnergy = 11.2
        fig = visualizeEnergies(weights, energies)
        self.assertTrue(fig is not None)
        
        fig = visualizeEnergies(weights, energies, showAsBars=True, signalEnergy=signalEnergy)
        self.assertTrue(fig is not None)
        
    def test_visualizeInformationRatesOptimality(self):
        scales = [32,64,128]
        sparseInfoRates = [20, 10, 5.5]
        
        scalesRef = [32,48,128]
        optimalInfoRates = [15, 8.4, 2.5]
        
        fig = visualizeInformationRatesOptimality(scales, sparseInfoRates, scalesRef, optimalInfoRates)
        self.assertTrue(fig is not None)
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    np.seterr(all='raise')
    unittest.main()
