
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
import ctypes
import logging
import collections
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from hsc.dataset import MultilevelDictionary, scalesToWindowSizes, addSingletonBases
from hsc.modeling import HierarchicalConvolutionalMatchingPursuit, HierarchicalConvolutionalSparseCoder, ConvolutionalDictionaryLearner
from hsc.analysis import calculateEmpiricalInformationRates
from hsc.utils import findGridSize, profileFunction

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    # Fix seed for random number generation for reproducible results
    np.random.seed(42)

    # NOTE: set so that any numerical error will raise an exception
    np.seterr(all='raise')

    logging.basicConfig(level=logging.INFO)
    cdir = os.path.dirname(os.path.realpath(__file__))
    outputResultPath = os.path.join(cdir, 'mlcsc')
    if not os.path.exists(outputResultPath):
        os.makedirs(outputResultPath)
    
    # Load reference multilevel dictionary from file
    filePath = os.path.join(cdir, 'multilevel-dict.pkl')
    multilevelDict = MultilevelDictionary.restore(filePath)
    
    # Load the reference training signal from file
    filePath = os.path.join(cdir, 'dataset-train.npz')
    trainData = np.load(filePath)
    trainSignal = trainData['signal']
    trainEvents = trainData['events']
    logger.info('Number of samples in dataset (training): %d' % (len(trainSignal)))
    logger.info('Number of events in dataset (training): %d' % (len(trainEvents)))
    
    # Load the reference testing signal from file
    filePath = os.path.join(cdir, 'dataset-test.npz')
    testData = np.load(filePath)
    testSignal = testData['signal']
    testEvents = testData['events']
    logger.info('Number of samples in dataset (testing): %d' % (len(testSignal)))
    logger.info('Number of events in dataset (testing): %d' % (len(testEvents)))

    # Reduce the size of the training and testing data
    nbSamples = 20000
    trainSignal = trainSignal[:nbSamples]
    testSignal = testSignal[:nbSamples]
    
    # Learn multilevel dictionary on the training data
    dictionaries = []
    input = trainSignal
    counts = np.array([16, 32, 64])
    scales = np.array([32, 64, 96])
    snr = 10.0
    nbLevels = len(counts)
    widths = scalesToWindowSizes(scales)
    for level, k, windowSize in zip(range(nbLevels), counts, widths):
        
        logger.info('Learning dictionary (samples)...')
        cdl = ConvolutionalDictionaryLearner(k, windowSize, algorithm='samples', avoidSingletons=True)
        D = cdl.train(input)
        assert D.shape[0] == k
        dictionaries.append(D)
    
        logger.info('Encoding signal...')
        if level > 0:
            # NOTE: since the dictionaries for the higher levels were learned on expanded representations (with singletons), the later must be added explicitly
            #       before creating the multilevel dictionary
            multilevelDict = MultilevelDictionary.fromRawDictionaries(addSingletonBases(dictionaries), scales[:len(dictionaries)], hasSingletonBases=True)
        else:
            multilevelDict = MultilevelDictionary.fromRawDictionaries(dictionaries, scales[:len(dictionaries)])
            
        hcmp = HierarchicalConvolutionalMatchingPursuit()
        hcsc = HierarchicalConvolutionalSparseCoder(multilevelDict, approximator=hcmp)
        
        # NOTE: for all levels but the last one, return the coefficients from the last level only, without redistributing the activations to lower levels
        if level == 0:
            coefficients, residual = hcsc.encode(trainSignal, toleranceSnr=snr, nbBlocks=100, alpha=0.0, singletonWeight=1.0, returnDistributed=False)
        elif level < nbLevels - 1:
            coefficients = hcsc.encodeFromLevel(coefficients, toleranceSnr=snr, nbBlocks=100, alpha=0.0, singletonWeight=1.0, returnDistributed=False)
        input = coefficients[-1].todense()
 
    # Visualize dictionary and save to disk as images
    logger.info('Generating dictionary visualizations...')
    figs = multilevelDict.visualize(maxCounts=16)
    for l,fig in enumerate(figs):
        fig.savefig(os.path.join(outputResultPath, 'dict-l%d.eps' % (l)), format='eps', dpi=1200)
    
    # Save multi-level dictionary to disk, as the reference
    filePath = os.path.join(outputResultPath, 'multilevel-dict.pkl')
    logger.info('Saving dictionary to file: %s' % (filePath))
    multilevelDict.save(filePath)
    
    # Analyze encoding on the test signal
    logger.info('Analyzing encoding...')
    hcmp = HierarchicalConvolutionalMatchingPursuit()
    hcsc = HierarchicalConvolutionalSparseCoder(multilevelDict, approximator=hcmp)
    coefficients, residual = hcsc.encode(testSignal, toleranceSnr=snr, nbBlocks=100, alpha=0.0, singletonWeight=1.0, returnDistributed=True)
    nbTotalCoefficients = np.sum([c.nnz for c in coefficients])
    logger.info('Total number of coefficients: %d' % (nbTotalCoefficients))
    for level, c in enumerate(coefficients):
        logger.info('Number of coefficients at level %d: %d (%4.1f%%)' % (level,c.nnz, float(c.nnz)/nbTotalCoefficients*100.0))
    
    rawInfoRate, sparseInfoRate = calculateEmpiricalInformationRates(testSignal, coefficients, multilevelDict)
    logger.info('Bitrate before encoding: %f bit/sample' % (rawInfoRate))
    logger.info('Bitrate after encoding: %f bit/sample' % (sparseInfoRate))
    logger.info('Compression ratio (raw/encoded): %f' % (rawInfoRate/sparseInfoRate))
    
    logger.info('All done.')
    