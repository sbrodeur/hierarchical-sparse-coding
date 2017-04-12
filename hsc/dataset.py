
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
import cPickle as pickle
import collections
import logging
import numpy as np
import scipy
import scipy.sparse
import matplotlib
import matplotlib.pyplot as plt

from hsc.analysis import calculateBitForDatatype, calculateMultilevelInformationRates
from hsc.utils import findGridSize, overlapAdd, normalize

logger = logging.getLogger(__name__)

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

class MultilevelDecompositionException(Exception):
    pass

class MultilevelDictionary(object):
    
    def __init__(self, dictionaries, scales, representations, decompositions, hasSingletonBases=False):
        assert len(dictionaries) > 0
        assert len(scales) > 0
        
        self.dictionaries = dictionaries
        self.scales = scales
        self.representations = representations
        self.decompositions = decompositions
        self.hasSingletonBases = hasSingletonBases
        
        if decompositions is not None:
            self.counts = np.array( [dictionaries[0].shape[0]] + [len(decomposition) for decomposition in decompositions], np.int)
        else:
            self.counts = np.array([dictionary.shape[0] for dictionary in dictionaries], dtype=np.int)

        if self.hasSingletonBases:
            countsNoSingletons = [self.counts[0],]
            for level in range(1, len(self.counts)):
                nbSingletonsLevel = np.sum(countsNoSingletons)
                nbStandardFeaturesLevel = self.counts[level] - nbSingletonsLevel
                countsNoSingletons.append(nbStandardFeaturesLevel)
            self.countsNoSingletons = np.array(countsNoSingletons, dtype=np.int)
        else:
            self.countsNoSingletons = np.copy(self.counts)

    @classmethod
    def fromRawDictionaries(cls, dictionaries, scales, hasSingletonBases=False):
        assert len(dictionaries) > 0
        assert len(scales) > 0
        
        # Calculate decompositions and representations
        # Loop for all levels
        decompositions = []
        representations = [dictionaries[0],]
        widths = scalesToWindowSizes(scales)
        for level, dictionary in enumerate(dictionaries):
            assert dictionary.shape[1] == widths[level]
            if level == 0:
                continue
            
            # Loop for all patterns in the dictionary at current level
            levelDecompositions = []
            levelRepresentations = []
            for pattern in dictionary:
                nnzMask = np.where(np.abs(pattern) > 0.0)
                if len(nnzMask[0]) > 0:
                    coefficients = pattern[nnzMask]
                    positions, selectedIndices = nnzMask[0], nnzMask[1]
                    
                    # Convert level-relative positions to input-level positions
                    if np.mod(scales[level-1], 2) == 0:
                        # Even
                        offset = scales[level-1]/2-1
                    else:
                        # Odd
                        offset = scales[level-1]/2
                    positions += offset
                    
                    selectedLevels = (level-1) * np.ones_like(positions, dtype=np.int32)
                    levelDecompositions.append([selectedLevels, selectedIndices, positions, coefficients])
                    assert len(coefficients) > 0
                    
                    # Compose each pattern linearly
                    signal = np.zeros(scales[level], dtype=dictionary.dtype)
                    for l, i, t, c in zip(selectedLevels, selectedIndices, positions, coefficients):
                        # Additive overlap to the signal, taking into account boundaries
                        overlapAdd(signal, element=c*representations[l][i,:], t=t, copy=False)
                            
                    # Normalize l2-norm
                    l2norm = np.sqrt(np.sum(np.square(signal)))
                    assert l2norm > 0.0
                    signal /= l2norm
                    
                    levelRepresentations.append(signal)
                else:
                    levelDecompositions.append([])
                    levelRepresentations.append(np.zeros(scales[level], dtype=dictionary.dtype))
                    logger.warn('Null pattern found in dictionary at level %d' % (level))
        
            decompositions.append(levelDecompositions)
            representations.append(np.stack(levelRepresentations))
        
        return cls(dictionaries, scales, representations, decompositions, hasSingletonBases)

    @classmethod
    def fromBaseDictionary(cls, baseDict):
        assert len(baseDict) > 0
        return cls(dictionaries=[baseDict,], 
                   scales=[baseDict.shape[1],], 
                   representations=[baseDict,],
                   decompositions=None)

    @classmethod
    def fromDecompositions(cls, baseDict, decompositions, scales, hasSingletonBases=False):
        assert decompositions is not None and len(decompositions) > 0
        assert len(scales) > 0
        
        # Detect if multilevel decomposition
        isMultilevel = False
        for level in range(1, len(scales)):
            for selectedLevels, selectedIndices, positions, coefficients in decompositions[level-1]:
                if not np.array_equal(selectedLevels, (level-1) * np.ones_like(selectedLevels)):
                    isMultilevel = True
                    break
            if isMultilevel:
                break
        
        logger.debug('Precomputing raw dictionaries and representations from base dictionary and decompositions...')
        if isMultilevel:
            # Loop over all higher levels
            dictionaries = [baseDict,]
            counts = np.array( [dictionaries[0].shape[0]] + [len(decomposition) for decomposition in decompositions], np.int)
            for level in range(1, len(scales)):
                bases = np.zeros((counts[level], scales[level], counts[level-1]), dtype=baseDict.dtype)
                dictionaries.append(bases)
            dictionaries = addSingletonBases(dictionaries)
            hasSingletonBases = True
            
            for level in range(1, len(scales)):
            
                # Loop over all patterns to compose into dense coefficient vector
                bases = []
                for selectedLevels, selectedIndices, positions, coefficients in decompositions[level-1]:
                    
                    # Normalize
                    coefficients /= np.sqrt(np.sum(np.square(coefficients)))
                    
                    for l, fIdx, position, c in zip(selectedLevels, selectedIndices, positions, coefficients):
                    
                        # Convert input-level position to level-relative position
                        if np.mod(scales[l], 2) == 0:
                            # Even
                            offset = scales[l]/2-1
                        else:
                            # Odd
                            offset = scales[l]/2
                        
                        # Passthrough offset
                        fIdxOffset = np.sum(counts[:l])
                        
                        dictionaries[level][fIdxOffset + fIdx, position - offset] = c
                
        else:
            # Loop over all higher levels
            dictionaries = [baseDict,]
            widths = scalesToWindowSizes(scales)
            counts = np.array( [dictionaries[0].shape[0]] + [len(decomposition) for decomposition in decompositions], np.int)
            for level in range(1, len(scales)):
            
                # Loop over all patterns to compose into dense coefficient vector
                bases = []
                for selectedLevels, selectedIndices, positions, coefficients in decompositions[level-1]:
                    
                    # Compose each pattern
                    basis = np.zeros((widths[level], counts[level-1]), dtype=coefficients.dtype)
    
                    # Convert input-level positions to level-relative positions
                    if np.mod(scales[level-1], 2) == 0:
                        # Even
                        offset = scales[level-1]/2-1
                    else:
                        # Odd
                        offset = scales[level-1]/2
                    basis[positions - offset, selectedIndices] = coefficients
                    
                    # Normalize
                    basis /= np.sqrt(np.sum(np.square(basis)))
                    
                    bases.append(basis)
                    
                bases = np.stack(bases)
                dictionaries.append(bases)
        
        # Loop over all levels
        logger.debug('Precomputing input-level representations from base dictionary and decompositions...')
        representations = [baseDict,]
        for scale, levelDecompositions in zip(scales[1:], decompositions):
        
            # Loop over all patterns to compose into a discrete signal at current level
            levelRepresentations = []
            for selectedLevels, selectedIndices, positions, coefficients in levelDecompositions:
                
                # Compose each pattern linearly
                signal = np.zeros(scale, dtype=baseDict.dtype)
                for l, i, t, c in zip(selectedLevels, selectedIndices, positions, coefficients):
                    # Additive overlap to the signal, taking into account boundaries
                    overlapAdd(signal, element=c*representations[l][i,:], t=t, copy=False)
                        
                # Normalize l2-norm
                signal /= np.sqrt(np.sum(np.square(signal)))
                
                levelRepresentations.append(signal)
                
            representations.append(np.stack(levelRepresentations))
        
        return cls(dictionaries, scales, representations, decompositions, hasSingletonBases)

    def upToLevel(self, level):
        assert level < self.getNbLevels()
        if level == 0:
            instance = MultilevelDictionary.fromRawDictionaries(self.dictionaries[:1], self.scales[:1], self.hasSingletonBases)
        else:
            instance = MultilevelDictionary.fromDecompositions(self.dictionaries[0], self.decompositions[:level], self.scales[:level+1], self.hasSingletonBases)
        return instance
    
    def withSingletonBases(self):
        
        if not self.hasSingletonBases:
            if self.getNbLevels() > 1:
                # Create a new instance with expanded dictionaries
                newDictionaries = addSingletonBases(self.dictionaries)
                instance = MultilevelDictionary.fromRawDictionaries(newDictionaries, self.scales, hasSingletonBases=True)
            else:
                logger.warn('Could not add expanded bases since there is only one level')
                instance = self
        else:
            logger.warn('Could not add expanded bases since they already exist')
            instance = self

        return instance

    def visualize(self, maxCounts=None, shuffle=True, annotate=False):
        figs = []
        for level in range(self.getNbLevels()):
            fig = plt.figure(figsize=(8,8), facecolor='white', frameon=True)
            fig.canvas.set_window_title('Level %d dictionary' % (level))
            fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                                hspace=0.01, wspace=0.01)
            
            levelRepresentations = self.representations[level]
            if self.hasSingletonBases:
                # Filter out representations of singletons
                l0norms = np.sum(self.dictionaries[level] != 0.0, axis=tuple(range(1, self.dictionaries[level].ndim)))
                levelRepresentations = levelRepresentations[np.where(l0norms > 1)]
            count = levelRepresentations.shape[0]
            
            if maxCounts is not None:
                if isinstance(maxCounts, collections.Iterable):
                    count = min(count, int(maxCounts[level]))
                else:
                    count = min(count, int(maxCounts))
                
            m,n = findGridSize(count)
            
            indices = np.arange(levelRepresentations.shape[0], dtype=np.int)
            if shuffle:
                indices = np.random.permutation(indices)
            indices = indices[:count]
            idx = 0
            for i in range(m):
                for j in range(n):
                    ax = fig.add_subplot(m,n,idx+1)
                    ax.plot(levelRepresentations[indices[idx]], linewidth=2, color='k')
                    r = np.max(np.abs(levelRepresentations[indices[idx]]))
                    ax.set_ylim(-r, r)
                    ax.set_axis_off()
                    if annotate:
                        ax.annotate(str(indices[idx]), xy=(r/2,r/2))
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
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise Exception('Unsupported format: %s' % (format))

    def getNbLevels(self):
        return len(self.scales)
    
    def getRawDictionary(self, level):
        assert level >= 0 and level < self.getNbLevels()
        return self.dictionaries[level]

    def getBaseDictionary(self):
        return self.dictionaries[0]

    def getMultiscaleDictionaries(self):
        # Return the precomputed representations at the input level
        return self.representations

class MultilevelDictionaryGenerator(object):
    
    def __init__(self):
        pass

    def generate(self, scales, counts, decompositionSize=4, positionSampling='random', weightSampling='random', multilevelDecomposition=True, maxNbPatternsConsecutiveRejected=100, nonNegativity=False):
        assert len(scales) > 0
        assert len(counts) > 0
        assert len(scales) == len(counts)
        
        scales = np.array(scales, np.int)
        counts = np.array(counts, np.int)
        
        # Base-level dictionary
        logger.debug('Generating base dictionary...')
        baseDict = self._generateBaseDictionary(counts[0], scales[0], maxNbPatternsConsecutiveRejected, nonNegativity)
        
        # High-level dictionaries
        logger.debug('Generating high-level dictionaries...')
        if len(counts) > 1:
            decompositions = self._generateHighLevelDecompositions(baseDict, scales, counts, decompositionSize, positionSampling, weightSampling, multilevelDecomposition, maxNbPatternsConsecutiveRejected)
            multilevelDict = MultilevelDictionary.fromDecompositions(baseDict, decompositions, scales)
        else:
            multilevelDict = MultilevelDictionary.fromBaseDictionary(baseDict)
            
        return multilevelDict 
            
    def _generateBaseDictionary(self, nbPatterns, nbPoints, maxNbPatternsConsecutiveRejected=100, nonNegativity=False):
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
        patterns = np.zeros((nbPatterns, nbPoints), dtype=np.float32)
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
            
            # Non-negativity constraint
            if nonNegativity:
                y = np.abs(y)
            
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

    def _generateHighLevelDecompositions(self, baseDict, scales, counts, decompositionSizes, positionSampling='random', weightSampling='random', multilevelDecomposition=True, maxNbPatternsConsecutiveRejected=100):
        assert maxNbPatternsConsecutiveRejected > 0
        
        representations = [baseDict,]
        decompositions = []
        for level, count in enumerate(counts):
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
            patterns = np.zeros((count, scales[level]), dtype=np.float32)
            while (nbPatternsFound < count):
            
                # Random sampling of the lower-level activated patterns
                if multilevelDecomposition:
                    # Sample from all previous levels
                    randomLevels = np.random.randint(low=0, high=level, size=decompositionSize)
                else:
                    # Sample from previous level only
                    randomLevels = (level-1) * np.ones((decompositionSize,), dtype=np.int)
                
                # NOTE: make sure the same pattern (at a particular level) is not selected twice
                selectedLevels = []
                selectedIndices = []
                for l in range(level):
                    nbSamplesFromLevel = len(np.where(randomLevels == l)[0])
                    if nbSamplesFromLevel > 0:
                        if nbSamplesFromLevel > counts[l]:
                            raise Exception('Unable to decompose to %d items at sublevels: dictionary size at level %d is too low (%d)' % (nbSamplesFromLevel, l, counts[l]))
                        indices = np.random.permutation(counts[l]).astype(dtype=np.int)
                        selectedLevels.append(l * np.ones((nbSamplesFromLevel,), dtype=np.int))
                        selectedIndices.append(indices[:nbSamplesFromLevel])
                selectedLevels = np.concatenate(selectedLevels)
                selectedIndices = np.concatenate(selectedIndices)
                
                # Random sampling of the time positions of the lower-level activated patterns
                while True:
                    if positionSampling == 'random':
                        positions = []
                        for selectedLevel in selectedLevels:
                            if np.mod(scales[selectedLevel], 2) == 0:
                                # Even scale
                                position = np.random.randint(low=scales[selectedLevel]/2-1,
                                                             high=scales[level] - scales[selectedLevel]/2)
                            else:
                                # Odd scale
                                position = np.random.randint(low=scales[selectedLevel]/2,
                                                             high=scales[level] - scales[selectedLevel]/2)
                            positions.append(position)
                        positions = np.array(positions, dtype=np.int)
                        
                    elif positionSampling == 'no-overlap':
                        
                        totalConcatLength = np.sum([scales[selectedLevel] for selectedLevel in selectedLevels])
                        remainingLength = scales[level] - totalConcatLength
                        assert remainingLength >= 0
                        
                        positions = []
                        nextMinPosition = 0.0
                        for selectedLevel in selectedLevels:
                            if np.mod(scales[selectedLevel], 2) == 0:
                                # Even scale
                                position = nextMinPosition + np.random.randint(low=scales[selectedLevel]/2-1,
                                                                               high=scales[selectedLevel]/2 + remainingLength)
                                remainingLength -= (position - nextMinPosition - (scales[selectedLevel]/2-1))
                                nextMinPosition = position + scales[selectedLevel]/2
                            else:
                                # Odd scale
                                position = nextMinPosition + np.random.randint(low=scales[selectedLevel]/2,
                                                                               high=scales[selectedLevel]/2 + remainingLength + 1)
                                remainingLength -= (position - nextMinPosition - (scales[selectedLevel]/2))
                                nextMinPosition = position + scales[selectedLevel]/2
                            assert remainingLength >= 0
                            
                            positions.append(position)
                        positions = np.array(positions, dtype=np.int)
                    else:
                        raise Exception('Unsupported position sampling method: %s' % (positionSampling))
                    
                    # Make sure that the positions cover the whole range, and are not too densely located
                    r = float(np.max(positions) - np.min(positions))
                    if r >= 0.45 * scales[level] or decompositionSize == 1:
                        # Position solution found with enough span
                        break
                
                # Random sampling of the relative coefficients, then normalization
                if weightSampling == 'random':
                    coefficients = np.random.uniform(low=0.25, high=1.0, size=decompositionSize).astype(dtype=np.float32)
                elif weightSampling == 'constant':
                    coefficients = np.ones(decompositionSize, dtype=np.float32)
                else:
                    raise Exception('Unsupported weight sampling method: %s' % (weightSampling))
                coefficients /= np.sqrt(np.sum(np.square(coefficients)))
                
                nbPatternsSampled += 1
                
                signal = self._composePattern(level, scales, representations, selectedLevels, selectedIndices, positions, coefficients)

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
            levelRepresentations = []
            for selectedLevels, selectedIndices, positions, coefficients in levelDecompositions:
                signal = self._composePattern(level, scales, representations, selectedLevels, selectedIndices, positions, coefficients)
                levelRepresentations.append(signal)
                
            representations.append(np.array(levelRepresentations, dtype=np.float32))
            decompositions.append(levelDecompositions)
            
        return decompositions

    def _composePattern(self, level, scales, representations, selectedLevels, selectedIndices, positions, coefficients):
        signal = np.zeros(scales[level], dtype=coefficients.dtype)
        for l, i, t, c in zip(selectedLevels, selectedIndices, positions, coefficients):
            # Additive overlap to the signal, taking into account boundaries
            overlapAdd(signal, element=c*representations[l][i,:], t=t, copy=False)
                
        # Normalize l2-norm
        signal /= np.sqrt(np.sum(np.square(signal)))
        return signal

class SignalGenerator(object):

    def __init__(self, multilevelDict, rates):
        assert len(rates) == multilevelDict.getNbLevels()
        
        self.multilevelDict = multilevelDict
        self.rates = rates

    def _estimateOptimalRates(self, minimumCompressionRatio, nbSamples):
        
        dtype = self.multilevelDict.dictionaries[0].dtype
        c_bits = calculateBitForDatatype(dtype)
        factors = np.linspace(1e-6, 1.0, num=1000)[::-1]
        factorIdx = 0
        while True:
            scaledRates = np.copy(self.rates) * factors[factorIdx]
            avgInfoRates = calculateMultilevelInformationRates(self.multilevelDict, scaledRates, nbSamples, dtype=dtype)
            if avgInfoRates[0] <= c_bits * minimumCompressionRatio:
                # Valid rates found
                break
            else:
                factorIdx += 1
                if factorIdx >= len(factors):
                    raise Exception("Unable to find the optimal rates: initial rates are too high")
                logger.debug('Rates are too high (bitrate of %f bit/sample at first level): scaling so that maximum rate across levels is %f' % (avgInfoRates[0], np.max(scaledRates)))
        logger.info('Optimal rate scale found: %4.8f (for bitrate of %f bit/sample at first level)' % (np.max(scaledRates), avgInfoRates[0]))
        
        return scaledRates

    def generateEvents(self, nbSamples=1000, minimumCompressionRatio=None):
        
        if minimumCompressionRatio is not None:
            rates = self._estimateOptimalRates(minimumCompressionRatio, nbSamples)
        else:
            rates = self.rates
        
        events = []
        
        # Loop for each scale
        dtype = self.multilevelDict.dictionaries[0].dtype
        for level, scale in enumerate(self.multilevelDict.scales):
        
            levelRates = rates[level]
        
            # Loop for each pattern
            for i, pattern in enumerate(self.multilevelDict.representations[level]):
        
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
                coefficients = np.random.uniform(low=0.25, high=4.0, size=len(times)).astype(dtype)
                
                levels = level * np.ones_like(times, dtype=np.int)
                indices = i * np.ones_like(times, dtype=np.int)
                events.extend([event for event in zip(times, levels, indices, coefficients)])
                
        # Sort events by time (increasing)
        events = sorted(events, key=lambda x: x[0])
                
        # Convert to mixed-type numpy array
        events = np.array(events, dtype=('int32,int32,int32,float32'))
        
        if minimumCompressionRatio is not None:
            return events, rates
        else:
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
        dtype = self.multilevelDict.dictionaries[0].dtype
        signal = np.zeros(nbSamples, dtype=dtype)
        for t, l, i, c in events:
            # Additive overlap to the signal, taking into account boundaries
            overlapAdd(signal, element=c*self.multilevelDict.representations[l][i,:], t=t, copy=False)
        
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
    
def convertSparseMatricesToEvents(coefficients):
    
    events = []
    for level, c in enumerate(coefficients):
        c = c.tocoo()
        events.extend([event for event in zip(c.row, level*np.ones_like(c.row), c.col, c.data)])
    
    # Sort events by time (increasing)
    events = sorted(events, key=lambda x: x[0])
            
    # Convert to mixed-type numpy array
    events = np.array(events, dtype=('int32,int32,int32,float32'))
    
    return events
    
def convertEventsToSparseMatrices(events, counts, sequenceLength):
    
    tIndices = np.array([event[0] for event in events], dtype=np.int)
    levels = np.array([event[1] for event in events], dtype=np.int)
    fIndices = np.array([event[2] for event in events], dtype=np.int)
    cValues = np.array([event[3] for event in events], dtype=events.dtype[-1])
    
    coefficients = []
    for level, count in enumerate(counts):
        mask = np.where(levels == level)
        coefficients.append(scipy.sparse.coo_matrix((cValues[mask], (tIndices[mask], fIndices[mask])), shape=(sequenceLength, count)).tocsr())
    return coefficients
    
def addSingletonBases(dictionaries):
    assert len(dictionaries) > 1
    
    # Loop over all higher levels
    newDictionaries = [dictionaries[0], ]
    newCounts = [dictionaries[0].shape[0],]
    for level in range(1, len(dictionaries)):
        D = dictionaries[level]
        
        if level > 1:
            # Expand dictionary on the feature axis to cover extra singleton at previous level
            nbInputFeatures = newDictionaries[level-1].shape[0]
            D = np.pad(D, [(0,0),(0,0),(nbInputFeatures - D.shape[-1],0)], mode='constant')
        assert D.shape[-1] == newDictionaries[level-1].shape[0]
        
        # Expand dictionary with extra singleton bases
        nbSingletons = newCounts[level-1]
        singletons = np.zeros((nbSingletons,) + D.shape[1:], dtype=D.dtype)
        indices = np.arange(nbSingletons)
        
        # NOTE: the singleton is already normalized
        filterWidth = D.shape[1]
        if np.mod(filterWidth, 2) == 0:
            # Even size
            offset = filterWidth/2-1
        else:
            # Odd size
            offset = filterWidth/2
        singletons[indices, offset, indices] = 1.0
        
        newD = np.concatenate((singletons, D), axis=0)
        newDictionaries.append(newD)
        newCounts.append(newD.shape[0])

    return newDictionaries
    
def scalesToWindowSizes(scales):
    assert len(scales) > 0
    
    windowSizes = [scales[0],]
    for level in range(1,len(scales)):
        width = int(scales[level] - scales[level-1] + 1)
        windowSizes.append(width)
    return np.array(windowSizes, dtype=np.int)
