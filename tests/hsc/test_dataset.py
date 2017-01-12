
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
import unittest
import tempfile
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from hsc.dataset import Perlin, MultilevelDictionary, SignalGenerator

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

    def test_init(self):
        multilevelDict = MultilevelDictionary(scales=[32], counts=[8], maxNbPatternsConsecutiveRejected=100)
        self.assertTrue(len(multilevelDict.dicts) == 1)
        self.assertTrue(multilevelDict.dicts[0].shape == (8,32))
        
        multilevelDict = MultilevelDictionary(scales=[63], counts=[7], maxNbPatternsConsecutiveRejected=100)
        self.assertTrue(len(multilevelDict.dicts) == 1)
        self.assertTrue(multilevelDict.dicts[0].shape == (7,63))
        
        multilevelDict = MultilevelDictionary(scales=[32, 63], counts=[8,15],
                                              decompositionSize=4, maxNbPatternsConsecutiveRejected=100)
        self.assertTrue(len(multilevelDict.dicts) == 2)
        self.assertTrue(multilevelDict.dicts[0].shape == (8,32))
        self.assertTrue(multilevelDict.dicts[1].shape == (15,63))
        
        multilevelDict = MultilevelDictionary(scales=[32, 63], counts=[8,15],
                                              decompositionSize=1, maxNbPatternsConsecutiveRejected=100)
        self.assertTrue(len(multilevelDict.dicts) == 2)
        self.assertTrue(multilevelDict.dicts[0].shape == (8,32))
        self.assertTrue(multilevelDict.dicts[1].shape == (15,63))
        
    def test_visualize(self):
        multilevelDict = MultilevelDictionary(scales=[32], counts=[8],
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
        
        multilevelDict = MultilevelDictionary(scales=[32, 63], counts=[8,15],
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

            # Create and save dictionary
            multilevelDictRef = MultilevelDictionary(scales=[32,64], counts=[8,16])
            multilevelDictRef.save(path)
             
            # Restore dictionary
            multilevelDict = MultilevelDictionary.restore(path)
            for i in range(len(multilevelDict.dicts)):
                self.assertTrue(np.allclose(multilevelDict.dicts[i], multilevelDictRef.dicts[i], atol=1e-6))
            for decompositionRef, decomposition in zip(multilevelDictRef.decompositions, multilevelDict.decompositions):
                for (selectedLevelsRef, selectedIndicesRef, positionsRef, coefficientsRef), \
                    (selectedLevels, selectedIndices, positions, coefficients) in zip(decompositionRef, decomposition):
                    self.assertTrue(np.allclose(selectedLevelsRef, selectedLevels))
                    self.assertTrue(np.allclose(selectedIndicesRef, selectedIndices))
                    self.assertTrue(np.allclose(positionsRef, positions))
                    self.assertTrue(np.allclose(coefficientsRef, coefficients))
             
        finally:
            os.remove(path)
        
class TestSignalGenerator(unittest.TestCase):

    def test_init(self):
        multilevelDict = MultilevelDictionary(scales=[32], counts=[8], maxNbPatternsConsecutiveRejected=100)
        generator = SignalGenerator(multilevelDict, rates=[0.001])
        
    def test_generate_events(self):
        
        nbSamples = int(1e5)
        for nbPatterns in [1, 4]:
            rate = 0.1
            multilevelDict = MultilevelDictionary(scales=[32], counts=[nbPatterns], maxNbPatternsConsecutiveRejected=100)
            generator = SignalGenerator(multilevelDict, rates=[rate])
            events = generator.generateEvents(nbSamples)
            self.assertTrue(np.allclose(rate * nbPatterns, float(len(events))/nbSamples, rtol=0.1))

        nbSamples = int(1e5)
        for nbPatterns in [1, 4]:
            rate = 0.1
            multilevelDict = MultilevelDictionary(scales=[32,64], counts=[nbPatterns, nbPatterns], maxNbPatternsConsecutiveRejected=100)
            generator = SignalGenerator(multilevelDict, rates=[rate, rate])
            events = generator.generateEvents(nbSamples)
            self.assertTrue(np.allclose(rate * nbPatterns * len(multilevelDict.dicts), float(len(events))/nbSamples, rtol=0.1))
        
        nbSamples = int(1e4)
        nbPatterns = 4
        rate = 0.1
        multilevelDict = MultilevelDictionary(scales=[31,63], counts=[nbPatterns, nbPatterns], maxNbPatternsConsecutiveRejected=100)
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
          
    def test_generate_signal(self):
        
        nbSamples = int(1e4)
        nbPatterns = 4
        rate = 0.1
        multilevelDict = MultilevelDictionary(scales=[32], counts=[nbPatterns], maxNbPatternsConsecutiveRejected=100)
        generator = SignalGenerator(multilevelDict, rates=[rate])
        events = generator.generateEvents(nbSamples)
        signal = generator.generateSignalFromEvents(events, nbSamples=nbSamples)
        self.assertTrue(len(signal) == nbSamples)
        signal = generator.generateSignalFromEvents(events)
        self.assertTrue(np.allclose(len(signal), nbSamples, rtol=0.1))
        
        nbSamples = int(1e4)
        nbPatterns = 4
        rate = 0.1
        multilevelDict = MultilevelDictionary(scales=[31,63], counts=[nbPatterns, nbPatterns], maxNbPatternsConsecutiveRejected=100)
        generator = SignalGenerator(multilevelDict, rates=[rate, rate])
        signal = generator.generateSignalFromEvents(events, nbSamples=nbSamples)
        self.assertTrue(len(signal) == nbSamples)
        signal = generator.generateSignalFromEvents(events)
        self.assertTrue(np.allclose(len(signal), nbSamples, rtol=0.1))
          
if __name__ == '__main__':
    unittest.main()
