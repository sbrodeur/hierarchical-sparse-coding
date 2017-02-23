
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
import tempfile
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from hsc.dataset import Perlin, MultilevelDictionary, MultilevelDictionaryGenerator, SignalGenerator, scalesToWindowSizes, convertEventsToSparseMatrices

class TestPerlin(unittest.TestCase):

    def test_sample(self):
        
        perlin = Perlin()

        nbPoints = 512        
        span = 5.0
        x = np.arange(nbPoints) * span / nbPoints - 0.5 * span
        idx = np.random.randint(low=0, high=nbPoints)
        x = x[idx:idx+nbPoints]
        
        y = perlin.sample(x)
        self.assertTrue(len(x) == len(y))
        
        y = perlin.sample(x, octaves=5, persistence=0.75, lacunarity=1.0, repeat=512)
        self.assertTrue(len(x) == len(y))
        
    def test_shuffle(self):
        
        perlin = Perlin()

        nbPoints = 512        
        span = 5.0
        x = np.arange(nbPoints) * span / nbPoints - 0.5 * span
        idx = np.random.randint(low=0, high=nbPoints)
        x = x[idx:idx+nbPoints]
        
        y1 = perlin.sample(x)
        self.assertTrue(len(x) == len(y1))
        y2 = perlin.sample(x)
        self.assertTrue(np.allclose(y1,y2))
        
        perlin.shuffle()
        y3 = perlin.sample(x)
        self.assertFalse(np.allclose(y1,y3))
        self.assertFalse(np.allclose(y2,y3))

class TestMultilevelDictionary(unittest.TestCase):

    def test_fromRawDictionaries(self):
        
        mldg = MultilevelDictionaryGenerator()
        multilevelDictRef = mldg.generate(scales=[16, 32, 63], counts=[4, 8, 15],
                                          decompositionSize=2, multilevelDecomposition=False, 
                                          maxNbPatternsConsecutiveRejected=10)
            
        multilevelDict = MultilevelDictionary.fromRawDictionaries(multilevelDictRef.dictionaries, multilevelDictRef.scales)
        self.assertTrue(multilevelDict.getNbLevels() == multilevelDictRef.getNbLevels())
        for i in range(multilevelDict.getNbLevels()):
            self.assertTrue(np.allclose(multilevelDict.dictionaries[i], multilevelDictRef.dictionaries[i], atol=1e-6))
        for decompositionRef, decomposition in zip(multilevelDictRef.decompositions, multilevelDict.decompositions):
            for (selectedLevelsRef, selectedIndicesRef, positionsRef, coefficientsRef), \
                (selectedLevels, selectedIndices, positions, coefficients) in zip(decompositionRef, decomposition):
                self.assertTrue(np.allclose(np.sort(selectedLevelsRef), np.sort(selectedLevels)))
                self.assertTrue(np.allclose(np.sort(selectedIndicesRef), np.sort(selectedIndices)))
                if not np.allclose(np.sort(positionsRef), np.sort(positions)):
                    pass
                self.assertTrue(np.allclose(np.sort(positionsRef), np.sort(positions)))
                self.assertTrue(np.allclose(np.sort(coefficientsRef), np.sort(coefficients)))
        for i in range(multilevelDict.getNbLevels()):
            self.assertTrue(np.allclose(multilevelDict.representations[i], multilevelDictRef.representations[i], atol=1e-6))
        for i in range(multilevelDict.getNbLevels()):
            self.assertTrue(np.allclose(multilevelDict.getRawDictionary(i), multilevelDictRef.getRawDictionary(i), atol=1e-6))
        for i in range(multilevelDict.getNbLevels()):
            for base in multilevelDict.dictionaries[i]:
                self.assertTrue(np.allclose(np.sqrt(np.sum(np.square(base))), 1.0))
                    
    def test_fromDecompositions(self):
        
        for isMultilevel in [True, False]:
            mldg = MultilevelDictionaryGenerator()
            multilevelDictRef = mldg.generate(scales=[16, 32, 63], counts=[4, 8, 15],
                                              decompositionSize=2, multilevelDecomposition=isMultilevel, 
                                              maxNbPatternsConsecutiveRejected=10)
            
            multilevelDict = MultilevelDictionary.fromDecompositions(multilevelDictRef.getBaseDictionary(), multilevelDictRef.decompositions, multilevelDictRef.scales)
            self.assertTrue(multilevelDict.getNbLevels() == multilevelDictRef.getNbLevels())
            for decompositionRef, decomposition in zip(multilevelDictRef.decompositions, multilevelDict.decompositions):
                for (selectedLevelsRef, selectedIndicesRef, positionsRef, coefficientsRef), \
                    (selectedLevels, selectedIndices, positions, coefficients) in zip(decompositionRef, decomposition):
                    self.assertTrue(np.allclose(selectedLevelsRef, selectedLevels))
                    self.assertTrue(np.allclose(selectedIndicesRef, selectedIndices))
                    self.assertTrue(np.allclose(positionsRef, positions))
                    self.assertTrue(np.allclose(coefficientsRef, coefficients))
            for i in range(multilevelDict.getNbLevels()):
                self.assertTrue(np.allclose(multilevelDict.representations[i], multilevelDictRef.representations[i], atol=1e-6))
            for i in range(multilevelDict.getNbLevels()):
                self.assertTrue(np.allclose(multilevelDict.dictionaries[i], multilevelDictRef.dictionaries[i], atol=1e-6))
            
    def test_fromBaseDictionary(self):
        
        mldg = MultilevelDictionaryGenerator()
        multilevelDictRef = mldg.generate(scales=[16,], counts=[4,],
                                          maxNbPatternsConsecutiveRejected=10)
            
        multilevelDict = MultilevelDictionary.fromBaseDictionary(multilevelDictRef.getBaseDictionary())
        self.assertTrue(multilevelDict.getNbLevels() == multilevelDictRef.getNbLevels())
        for i in range(multilevelDict.getNbLevels()):
            self.assertTrue(np.allclose(multilevelDict.dictionaries[i], multilevelDictRef.dictionaries[i], atol=1e-6))
        self.assertTrue(multilevelDict.decompositions == None)
        for i in range(multilevelDict.getNbLevels()):
            self.assertTrue(np.allclose(multilevelDict.representations[i], multilevelDictRef.representations[i], atol=1e-6))
        for i in range(multilevelDict.getNbLevels()):
            self.assertTrue(np.allclose(multilevelDict.getRawDictionary(i), multilevelDictRef.getRawDictionary(i), atol=1e-6))
        
    def test_visualize(self):
        mldg = MultilevelDictionaryGenerator()
        multilevelDict = mldg.generate(scales=[32], counts=[8],
                                       maxNbPatternsConsecutiveRejected=100)
        figs = multilevelDict.visualize(maxCounts=9)
        self.assertTrue(len(figs) == 1)
        for fig in figs: plt.close(fig)
        figs = multilevelDict.visualize(maxCounts=[4])
        self.assertTrue(len(figs) == 1)
        for fig in figs: plt.close(fig)
        figs = multilevelDict.visualize()
        self.assertTrue(len(figs) == 1)
        for fig in figs: plt.close(fig)
        
        multilevelDict = mldg.generate(scales=[32, 63], counts=[8,15],
                                       decompositionSize=4, maxNbPatternsConsecutiveRejected=100)
        figs = multilevelDict.visualize(maxCounts=9)
        self.assertTrue(len(figs) == 2)
        for fig in figs: plt.close(fig)
        figs = multilevelDict.visualize(maxCounts=[9,65])
        self.assertTrue(len(figs) == 2)
        for fig in figs: plt.close(fig)
        figs = multilevelDict.visualize()
        self.assertTrue(len(figs) == 2)
        for fig in figs: plt.close(fig)
        
    def test_save_load(self):

        try:
            f, path = tempfile.mkstemp(suffix='.pkl')
            os.close(f)

            mldg = MultilevelDictionaryGenerator()

            # Create and save dictionary
            multilevelDictRef = mldg.generate(scales=[32,64], counts=[8,16])
            multilevelDictRef.save(path)
             
            # Restore dictionary
            multilevelDict = MultilevelDictionary.restore(path)
            for i in range(multilevelDict.getNbLevels()):
                self.assertTrue(np.allclose(multilevelDict.representations[i], multilevelDictRef.representations[i], atol=1e-6))
            for decompositionRef, decomposition in zip(multilevelDictRef.decompositions, multilevelDict.decompositions):
                for (selectedLevelsRef, selectedIndicesRef, positionsRef, coefficientsRef), \
                    (selectedLevels, selectedIndices, positions, coefficients) in zip(decompositionRef, decomposition):
                    self.assertTrue(np.allclose(selectedLevelsRef, selectedLevels))
                    self.assertTrue(np.allclose(selectedIndicesRef, selectedIndices))
                    self.assertTrue(np.allclose(positionsRef, positions))
                    self.assertTrue(np.allclose(coefficientsRef, coefficients))
             
        finally:
            os.remove(path)
        
    def test_withSingletonBases(self):
        
        mldg = MultilevelDictionaryGenerator()
        multilevelDict = mldg.generate(scales=[16, 32, 63], counts=[4, 8, 15],
                                       decompositionSize=2, multilevelDecomposition=False, 
                                       maxNbPatternsConsecutiveRejected=10)
        
        newMultilevelDict = multilevelDict.withSingletonBases()
        self.assertTrue(newMultilevelDict.getNbLevels() == multilevelDict.getNbLevels())
        self.assertTrue(np.array_equal(newMultilevelDict.counts, [4, 12, 27]))
        for level, count in zip(range(1, newMultilevelDict.getNbLevels()), [4, 8]):
            for base in newMultilevelDict.dictionaries[level][:count]:
                self.assertTrue(np.count_nonzero(base) == 1)
        for level, nbFeatures in zip(range(1, newMultilevelDict.getNbLevels()), [4, 12, 23]):
            self.assertTrue(newMultilevelDict.dictionaries[level].shape[-1] == nbFeatures)
        
class TestMultilevelDictionaryGenerator(unittest.TestCase):

    def test_generate(self):
        mldg = MultilevelDictionaryGenerator()
        
        multilevelDict = mldg.generate(scales=[32], counts=[8], maxNbPatternsConsecutiveRejected=100)
        self.assertTrue(multilevelDict.representations[0].shape == (8,32))
        self.assertTrue(multilevelDict.getNbLevels() == 1)
        
        multilevelDict = mldg.generate(scales=[63], counts=[7], maxNbPatternsConsecutiveRejected=100)
        self.assertTrue(multilevelDict.representations[0].shape == (7,63))
        self.assertTrue(multilevelDict.getNbLevels() == 1)
        
        multilevelDict = mldg.generate(scales=[32, 63], counts=[8,15],
                                       decompositionSize=4, maxNbPatternsConsecutiveRejected=100)
        self.assertTrue(multilevelDict.getNbLevels() == 2)
        self.assertTrue(multilevelDict.representations[0].shape == (8,32))
        self.assertTrue(multilevelDict.representations[1].shape == (15,63))
        
        multilevelDict = mldg.generate(scales=[32, 63], counts=[8,15],
                                       decompositionSize=1, maxNbPatternsConsecutiveRejected=100)
        self.assertTrue(multilevelDict.getNbLevels() == 2)
        self.assertTrue(multilevelDict.representations[0].shape == (8,32))
        self.assertTrue(multilevelDict.representations[1].shape == (15,63))
        
        
class TestSignalGenerator(unittest.TestCase):

    def test_init(self):
        mldg = MultilevelDictionaryGenerator()
        multilevelDict = mldg.generate(scales=[32], counts=[8], maxNbPatternsConsecutiveRejected=100)
        generator = SignalGenerator(multilevelDict, rates=[0.001])
        
    def test_generate_events(self):
        
        mldg = MultilevelDictionaryGenerator()
        
        nbSamples = int(1e5)
        for nbPatterns in [4, 7]:
            rate = 0.1
            multilevelDict = mldg.generate(scales=[32], counts=[nbPatterns], decompositionSize=2, maxNbPatternsConsecutiveRejected=100)
            generator = SignalGenerator(multilevelDict, rates=[rate])
            events = generator.generateEvents(nbSamples)
            self.assertTrue(np.allclose(rate * nbPatterns, float(len(events))/nbSamples, rtol=0.1))

        nbSamples = int(1e5)
        for nbPatterns in [4, 7]:
            rate = 0.1
            multilevelDict = mldg.generate(scales=[32,64], counts=[nbPatterns, nbPatterns], decompositionSize=2, maxNbPatternsConsecutiveRejected=100)
            generator = SignalGenerator(multilevelDict, rates=[rate, rate])
            events = generator.generateEvents(nbSamples)
            self.assertTrue(np.allclose(rate * nbPatterns * multilevelDict.getNbLevels(), float(len(events))/nbSamples, rtol=0.1))
        
        nbSamples = int(1e4)
        nbPatterns = 4
        rate = 0.1
        multilevelDict = mldg.generate(scales=[31,63], counts=[nbPatterns, nbPatterns], maxNbPatternsConsecutiveRejected=100)
        generator = SignalGenerator(multilevelDict, rates=[rate, rate])
        events = generator.generateEvents(nbSamples)
        times = [t for t,l,i,c in events]
        levels = [l for t,l,i,c in events]
        indices = [i for t,l,i,c in events]
        coefficients = [c for t,l,i,c in events]
        self.assertTrue(np.min(times) >= 0)
        self.assertTrue(np.max(times) < nbSamples)
        self.assertTrue(np.min(levels) == 0)
        self.assertTrue(np.max(levels) == 1)
        self.assertTrue(np.min(indices) == 0)
        self.assertTrue(np.max(indices) == nbPatterns-1)
        self.assertTrue(np.min(coefficients) > 0.25)
        self.assertTrue(np.max(coefficients) < 4.0)
          
    def test_generate_signal_rates(self):
        
        mldg = MultilevelDictionaryGenerator()
        
        nbSamples = int(1e4)
        nbPatterns = 4
        rate = 0.1
        multilevelDict = mldg.generate(scales=[32], counts=[nbPatterns], maxNbPatternsConsecutiveRejected=100)
        generator = SignalGenerator(multilevelDict, rates=[rate])
        events = generator.generateEvents(nbSamples)
        signal = generator.generateSignalFromEvents(events, nbSamples=nbSamples)
        self.assertTrue(len(signal) == nbSamples)
        signal = generator.generateSignalFromEvents(events)
        self.assertTrue(np.allclose(len(signal), nbSamples, rtol=0.1))
        
        nbSamples = int(1e4)
        nbPatterns = 4
        rate = 0.1
        multilevelDict = mldg.generate(scales=[31,63], counts=[nbPatterns, nbPatterns], maxNbPatternsConsecutiveRejected=100)
        generator = SignalGenerator(multilevelDict, rates=[rate, rate])
        events = generator.generateEvents(nbSamples)
        signal = generator.generateSignalFromEvents(events, nbSamples=nbSamples)
        self.assertTrue(len(signal) == nbSamples)
        signal = generator.generateSignalFromEvents(events)
        self.assertTrue(np.allclose(len(signal), nbSamples, rtol=0.1))
          
    def test_generate_signal_optimal(self):
        
        mldg = MultilevelDictionaryGenerator()
        
        nbSamples = int(1e4)
        nbPatterns = 4
        rate = 0.1
        multilevelDict = mldg.generate(scales=[31,63], counts=[nbPatterns, nbPatterns], maxNbPatternsConsecutiveRejected=100)
        generator = SignalGenerator(multilevelDict, rates=[rate, rate])
        events, rates = generator.generateEvents(nbSamples, minimumCompressionRatio=0.5)
        signal = generator.generateSignalFromEvents(events, nbSamples=nbSamples)
        self.assertTrue(len(signal) == nbSamples)
        signal = generator.generateSignalFromEvents(events)
        self.assertTrue(np.allclose(len(signal), nbSamples, rtol=0.1))
          
class TestFunctions(unittest.TestCase):
    
    def test_convertEventsToSparseMatrices(self):
        
        mldg = MultilevelDictionaryGenerator()
        
        nbSamples = int(1e4)
        rate = 0.1
        multilevelDict = mldg.generate(scales=[31,63], counts=[4, 7], maxNbPatternsConsecutiveRejected=100)
        generator = SignalGenerator(multilevelDict, rates=[rate, rate])
        events, _ = generator.generateEvents(nbSamples, minimumCompressionRatio=0.5)
        coefficients = convertEventsToSparseMatrices(events, multilevelDict.counts, nbSamples)
        self.assertTrue(np.array_equal(coefficients[0].shape, [nbSamples,4]))
        self.assertTrue(np.array_equal(coefficients[1].shape, [nbSamples,7]))
        self.assertTrue(int(np.sum([c.nnz for c in coefficients])) == len(events))
        eventLevels = np.array([event[1] for event in events], dtype=np.int)
        for level in range(multilevelDict.getNbLevels()):
            self.assertTrue(coefficients[level].nnz == np.count_nonzero(eventLevels == level))
            
        coefficients = [c.tocsr() for c in coefficients]
        for tIdx,level,fIdx,c in events:
            self.assertTrue(coefficients[level][tIdx,fIdx] == c)

    def test_scalesToWindowSizes(self):
        scales = [3,5,9]
        widths = scalesToWindowSizes(scales)
        self.assertTrue(len(widths) == len(scales))
        self.assertTrue(np.array_equal(widths, [3,3,5]))
        
        scales = [4,6,8]
        widths = scalesToWindowSizes(scales)
        self.assertTrue(len(widths) == len(scales))
        self.assertTrue(np.array_equal(widths, [4,3,3]))
          
        scales = [3,6,7]
        widths = scalesToWindowSizes(scales)
        self.assertTrue(len(widths) == len(scales))
        self.assertTrue(np.array_equal(widths, [3,4,2]))
        
        scales = [2,9,11]
        widths = scalesToWindowSizes(scales)
        self.assertTrue(len(widths) == len(scales))
        self.assertTrue(np.array_equal(widths, [2,8,3]))
        
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    np.seterr(all='raise')
    unittest.main()
