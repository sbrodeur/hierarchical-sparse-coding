
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
import pickle
import collections
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

from hsc.utils import findGridSize, overlapAdd

class Perlin(object):
    """
    Adapted from: https://github.com/caseman/noise/blob/master/_perlin.c
    """

    PERM = np.array([151,160,137,91,90,15,
            131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
            190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
            88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
            77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
            102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
            135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
            5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
            223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
            129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
            251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
            49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
            138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,151], dtype=np.int)

    def __init__(self):
        self.perm = np.copy(Perlin.PERM)

    def shuffle(self):
        np.random.shuffle(self.perm)

    def sample(self, x, octaves=1, persistence=0.5, lacunarity=2.0, repeat=1024, base=0):
        freq = 1.0
        amp = 1.0
        maximum = 0.0
        total = np.zeros_like(x)
        for i in range(octaves):
            total += self._noise(x * freq, int(repeat * freq), base) * amp
            maximum += amp
            freq *= lacunarity
            amp *= persistence
        noise = total / maximum

        return noise

    def _noise(self, x, repeat, base):
        i = np.mod(np.floor(x), repeat).astype(np.int)
        ii = np.mod(i + 1, repeat)
        i = (i & 255) + base
        ii = (ii & 255) + base

        x -= np.floor(x)
        fx = self._fade(x)

        return self._lerp(fx, self._grad(self.perm[i], x), self._grad(self.perm[ii], x-1)) * 0.4        

    def _fade(self, t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _lerp(self, t, a, b):
        return a + t * (b - a)

    def _grad(self, hash, x):
        g = (hash & 7) + 1.0
        g = np.where(hash & 8, -np.ones_like(g), g)
        return g * x

class MultilevelDictionary(object):
    
    def __init__(self, scales, counts, decompositionSize=4, positionSampling='random', maxNbPatternsConsecutiveRejected=100):
        assert len(scales) > 0
        assert len(counts) > 0
        assert len(scales) == len(counts)
        
        self.scales = np.array(scales, np.int)
        self.counts = np.array(counts, np.int)
        self.dicts = []
        
        # Base-level dictionary
        self.baseDict = self._generateBaseDictionary(counts[0], scales[0], maxNbPatternsConsecutiveRejected)
        
        # High-level dictionaries
        if len(self.counts) > 1:
            self.dicts, self.decompositions = self._generateHighLevelDictionaries(decompositionSize, positionSampling, maxNbPatternsConsecutiveRejected)
        else:
            self.dicts = [self.baseDict]

    def _generateBaseDictionary(self, nbPatterns, nbPoints, maxNbPatternsConsecutiveRejected=100):
        assert nbPatterns > 0
        assert nbPoints > 0
        assert maxNbPatternsConsecutiveRejected > 0
    
        logger.info('Generating base dictionary (l=0)...')
    
        # Perlin noise sampler
        perlin = Perlin()
        
        # Sampling loop, with adaptive rejection
        nbPatternsSampled = 0
        nbPatternsFound = 0
        nbPatternsRejected = 0
        nbPatternsConsecutiveRejected = 0
        maxCorrelations = np.linspace(0.05, 0.95, 64)
        maxCorrelationIdx = 0
        patterns = np.zeros((nbPatterns, nbPoints))
        while (nbPatternsFound < nbPatterns):
            
            # Generate the input vector twice as long, and extract a random window
            # NOTE: this is to avoid having the same zero crossing points across patterns
            span = 5.0
            maxOctaves = 3
            x = np.arange(2.0*nbPoints) * span / nbPoints - 0.5 * span
            idx = np.random.randint(low=0, high=nbPoints)
            x = x[idx:idx+nbPoints]
            
            # Randomly select the number of octaves and base
            octaves = np.random.randint(low=1, high=maxOctaves+1)
    
            # Shuffle the permutation array and sample a new sequence
            perlin.shuffle()
            y = perlin.sample(x, octaves)
            nbPatternsSampled += 1
            
            # Hanning windowing to make patterns localized in time and avoid discontinuties at endpoints
            window = np.hanning(nbPoints)
            y *= window
            
            # Normalize l2-norm
            y /= np.sqrt(np.sum(np.square(y)))
    
            if nbPatternsFound >= 1:
                # Find maximum dot product agains existing patterns
                c = np.max(np.abs(np.dot(patterns[:nbPatternsFound,:], y)))
                if c > maxCorrelations[maxCorrelationIdx]:
                    # Reject patterns because it is too similar
                    nbPatternsRejected += 1
                    nbPatternsConsecutiveRejected += 1
    
                    if nbPatternsConsecutiveRejected >= maxNbPatternsConsecutiveRejected:
                        if maxCorrelationIdx < len(maxCorrelations) - 1:
                            maxCorrelationIdx += 1 
                            logger.debug("Too much consecutively rejected patterns (%d rejected): maximum correlation threshold increased to %f" % (maxNbPatternsConsecutiveRejected, maxCorrelations[maxCorrelationIdx]))
                        else:
                            raise Exception("Unable to find the requested number of patterns: maximum correlation is too high")
                        nbPatternsConsecutiveRejected = 0
                        
                    continue # pragma: no cover
                
            patterns[nbPatternsFound,:] = y
            nbPatternsFound += 1
            nbPatternsConsecutiveRejected = 0
        
        logger.info("Number of patterns found = %d (%d rejected out of %d sampled)" % (nbPatternsFound, nbPatternsRejected, nbPatternsSampled))
    
        cd = np.abs(np.dot(patterns, patterns.T)) - np.diag(np.ones(patterns.shape[0]))
        logger.info("Maximum correlation between patterns = %f (%f average)" % (np.max(cd), np.mean(cd)))
    
        return patterns

    def _generateHighLevelDictionaries(self, decompositionSizes, positionSampling='random', maxNbPatternsConsecutiveRejected=100):
        assert maxNbPatternsConsecutiveRejected > 0
        
        dicts = [self.baseDict]
        decompositions = []
        for level, count in enumerate(self.counts):
            # Skip first level
            if level == 0:
                continue

            logger.info('Generating high-level dictionary (l=%d)...' % (level))

            levelDecompositions = []

            if isinstance(decompositionSizes, collections.Iterable):
                decompositionSize = decompositionSizes[level]
            else:
                decompositionSize = int(decompositionSizes)

            # Sampling loop, with adaptive rejection
            nbPatternsSampled = 0
            nbPatternsFound = 0
            nbPatternsRejected = 0
            nbPatternsConsecutiveRejected = 0
            maxCorrelations = np.linspace(0.05, 0.95, 64)
            maxCorrelationIdx = 0
            patterns = np.zeros((count, self.scales[level]))
            while (nbPatternsFound < count):
            
                # Random sampling of the lower-level activated patterns
                selectedLevels = np.random.randint(low=0, high=level, size=decompositionSize)
                selectedIndices = [np.random.randint(low=0, high=self.counts[selectedLevel]) for selectedLevel in selectedLevels]
                
                # Random sampling of the time positions of the lower-level activated patterns
                while True:
                    positions = []
                    for selectedLevel in selectedLevels:
                        
                        if positionSampling == 'random':
                            if np.mod(self.scales[selectedLevel], 2) == 0:
                                # Even scale
                                position = np.random.randint(low=self.scales[selectedLevel]/2-1,
                                                             high=self.scales[level] - self.scales[selectedLevel]/2)
                            else:
                                # Odd scale
                                position = np.random.randint(low=self.scales[selectedLevel]/2,
                                                             high=self.scales[level] - self.scales[selectedLevel]/2)
                        else:
                            raise Exception('Unsupported position sampling method: %s' % (positionSampling))
                        
                        positions.append(position)
                    positions = np.array(positions, dtype=np.int)
                    
                    # Make sure that the positions cover the whole range, and are not too densely located
                    r = float(np.max(positions) - np.min(positions))
                    if r >= 0.45 * self.scales[level] or decompositionSize == 1:
                        # Position solution found with enough span
                        break
                
                # Random sampling of the relative coefficients
                coefficients = np.random.uniform(low=0.25, high=1.0, size=decompositionSize)
                nbPatternsSampled += 1
                
                signal = self._composePattern(level, dicts, selectedLevels, selectedIndices, positions, coefficients)

                if nbPatternsFound >= 1:
                    # Find maximum dot product agains existing patterns
                    c = np.max(np.abs(np.dot(patterns[:nbPatternsFound,:], signal)))
                    if c > maxCorrelations[maxCorrelationIdx]:
                        # Reject patterns because it is too similar
                        nbPatternsRejected += 1
                        nbPatternsConsecutiveRejected += 1
        
                        if nbPatternsConsecutiveRejected >= maxNbPatternsConsecutiveRejected:
                            if maxCorrelationIdx < len(maxCorrelations) - 1:
                                maxCorrelationIdx += 1 
                                logger.debug("Too much consecutively rejected patterns (%d rejected): maximum correlation threshold increased to %f" % (maxNbPatternsConsecutiveRejected, maxCorrelations[maxCorrelationIdx]))
                            else:
                                raise Exception("Unable to find the requested number of patterns: maximum correlation is too high")
                            nbPatternsConsecutiveRejected = 0
                            
                        continue # pragma: no cover
                
                patterns[nbPatternsFound,:] = signal
                nbPatternsFound += 1
                nbPatternsConsecutiveRejected = 0
                levelDecompositions.append([selectedLevels, selectedIndices, positions, coefficients])

            logger.info("Number of patterns found = %d (%d rejected out of %d sampled)" % (nbPatternsFound, nbPatternsRejected, nbPatternsSampled))
            
            cd = np.abs(np.dot(patterns, patterns.T)) - np.diag(np.ones(patterns.shape[0]))
            logger.info("Maximum correlation between patterns = %f (%f average)" % (np.max(cd), np.mean(cd)))

            # Loop over all patterns to compose into a discrete signal
            levelDict = []
            for selectedLevels, selectedIndices, positions, coefficients in levelDecompositions:
                signal = self._composePattern(level, dicts, selectedLevels, selectedIndices, positions, coefficients)
                levelDict.append(signal)
            dicts.append(np.array(levelDict))
            decompositions.append(levelDecompositions)
            
        return dicts, decompositions

    def _composePattern(self, level, dicts, selectedLevels, selectedIndices, positions, coefficients):
        
        signal = np.zeros(self.scales[level])
        decompositionSize = len(selectedLevels)
        for l, i, t, c in zip(selectedLevels, selectedIndices, positions, coefficients):
            # Additive overlap to the signal, taking into account boundaries
            overlapAdd(signal, element=c*dicts[l][i,:], t=t, copy=False)
                
        # Normalize l2-norm
        signal /= np.sqrt(np.sum(np.square(signal)))
        return signal

    def visualize(self, maxCounts=None):
        figs = []
        for level, count in enumerate(self.counts):
            fig = plt.figure(figsize=(8,8), facecolor='white', frameon=True)
            fig.canvas.set_window_title('Level %d dictionary' % (level))
            fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                                hspace=0.01, wspace=0.01)
            
            if maxCounts is not None:
                if isinstance(maxCounts, collections.Iterable):
                    count = min(count, int(maxCounts[level]))
                else:
                    count = min(count, int(maxCounts))
                
            m,n = findGridSize(count)
            
            dict = self.dicts[level][:count]
            idx = 0
            for i in range(m):
                for j in range(n):
                    ax = fig.add_subplot(m,n,idx+1)
                    ax.plot(dict[idx], linewidth=2, color='k')
                    r = np.max(np.abs(dict[idx]))
                    ax.set_ylim(-r, r)
                    ax.set_axis_off()
                    idx += 1
                    if idx >= count:
                        break
        
            figs.append(fig)
        
        return figs

    @staticmethod
    def restore(filePath):
        filePath = os.path.abspath(filePath)
        _, format = os.path.splitext(filePath)
        if format == '.pkl' or format == '.p':
            with open(filePath, "rb" ) as f:
                instance = pickle.load(f)
        else:
            raise Exception('Unsupported format: %s' % (format))
        return instance
    
    def save(self, filePath):
        filePath = os.path.abspath(filePath)
        _, format = os.path.splitext(filePath)
        if format == '.pkl' or format == '.p':
            with open(filePath, "wb" ) as f:
                pickle.dump(self, f)
        else:
            raise Exception('Unsupported format: %s' % (format))

    def getNbLevels(self):
        return len(self.dicts)

class SignalGenerator(object):

    def __init__(self, multilevelDict, rates):
        assert len(rates) == multilevelDict.getNbLevels()
        
        self.multilevelDict = multilevelDict
        self.rates = rates

    def generateEvents(self, nbSamples=1000):
        
        events = []
        
        # Loop for each scale
        for level, scale in enumerate(self.multilevelDict.scales):
        
            levelRates = self.rates[level]
        
            # Loop for each pattern
            for i, pattern in enumerate(self.multilevelDict.dicts[level]):
        
                if isinstance(levelRates, collections.Iterable):
                    # Rate specified independently for each pattern
                    rate = levelRates[i]
                else:
                    # Rate fixed for all patterns of the current level
                    rate = float(levelRates)
        
                # Sample activation times
                times = self._generateSpikegram(rate, maxTime=nbSamples, continuousTime=False)
                if np.mod(scale, 2) == 0:
                    # Even scale
                    times = times[np.where((times >= np.floor(scale/2.0 - 1.0)) & (times <= nbSamples - np.floor(scale/2.0)))]
                else:
                    # Odd scale
                    times = times[np.where((times >= np.floor(scale/2.0)) & (times <= nbSamples - np.floor(scale/2.0)))]
                
                # Sample coefficients
                coefficients = np.random.uniform(low=0.25, high=4.0, size=len(times))
                
                levels = level * np.ones_like(times, dtype=np.int)
                indices = i * np.ones_like(times, dtype=np.int)
                coefficients = coefficients.astype(np.float32)
                events.extend([event for event in zip(times, levels, indices, coefficients)])
                
        # Sort events by time (increasing)
        events = sorted(events, key=lambda x: x[0])
                
        # Convert to mixed-type numpy array
        events = np.array(events, dtype=('int32,int32,int32,float32'))
                
        return events
                
    def generateSignalFromEvents(self, events, nbSamples=None):
        
        if nbSamples is None:
            # Estimate number of samples based on the events
            maxTime = 0
            for t, l, i, c in events:
                endTime = t + self.multilevelDict.scales[l]/2
                if endTime > maxTime:
                    maxTime = endTime
            nbSamples = maxTime
            logger.info('Number of samples estimated from the events: %d' % (nbSamples))
        
        # Compose discrete signal
        signal = np.zeros(nbSamples, dtype=np.float32)
        for t, l, i, c in events:
            # Additive overlap to the signal, taking into account boundaries
            overlapAdd(signal, element=c*self.multilevelDict.dicts[l][i,:], t=t, copy=False)
        
        return signal
                
    def _generateSpikegram(self, rate, maxTime=1.0, continuousTime=True):
        
        # Homogeneous Poisson process on continuous time
        t = 0.0
        times = []
        while t <= maxTime:
            t = t - np.log(np.random.uniform())/rate
            times.append(t)
            
        # Discard last event
        times = np.array(times[:-1])
        
        # Cast to discrete time, if specified
        if not continuousTime:
            times = np.unique(np.floor(times).astype(np.int))
        
        return times
