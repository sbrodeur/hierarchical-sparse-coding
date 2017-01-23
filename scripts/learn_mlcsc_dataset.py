
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
from hsc.analysis import calculateBitForDatatype
from hsc.utils import findGridSize, profileFunction

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    # Fix seed for random number generation for reproducible results
    np.random.seed(42)

    # NOTE: set so that any numerical error will raise an exception
    np.seterr(all='raise')

    logging.basicConfig(level=logging.INFO)
    cdir = os.path.dirname(os.path.realpath(__file__))
    
    # Load reference multilevel dictionary from file
    filePath = os.path.join(cdir, 'multilevel-dict.pkl')
    multilevelDict = MultilevelDictionary.restore(filePath)
    
    # Load the reference signal from file
    filePath = os.path.join(cdir, 'dataset-train.npz')
    trainData = np.load(filePath)
    trainSignal = trainData['signal']
    trainEvents = trainData['events']
    logger.info('Number of samples in dataset (training): %d' % (len(trainSignal)))
    logger.info('Number of events in dataset (training): %d' % (len(trainEvents)))
    
    # Load the reference signal from file
    filePath = os.path.join(cdir, 'dataset-test.npz')
    testData = np.load(filePath)
    testSignal = testData['signal']
    testEvents = testData['events']
    logger.info('Number of samples in dataset (testing): %d' % (len(testSignal)))
    logger.info('Number of events in dataset (testing): %d' % (len(testEvents)))

    testSignal = testSignal[:20000]
    
    fig = plt.figure(figsize=(8,4), facecolor='white', frameon=True)
    fig.canvas.set_window_title('Generated signal')
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.95,
                        hspace=0.01, wspace=0.01)
    ax = fig.add_subplot(111)
    n = np.arange(len(testSignal))
    ax.plot(n, testSignal, color='k')
    ax.set_xlim(0, len(testSignal))
    r = np.max(np.abs(testSignal))
    ax.set_ylim(-r, r)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Amplitude')
    #plt.show()
    
    dictionaries = []
    inputTrain = trainSignal
    
    counts = np.array([16, 32, 64])
    scales = np.array([32, 64, 96])
    snrs = np.array([40.0, 10.0, 10.0])
    nbLevels = len(counts)
    widths = scalesToWindowSizes(scales)
    for level, k, windowSize, snr in zip(range(nbLevels), counts, widths, snrs):
        
        logger.info('Learning dictionary (samples)...')
        cdl = ConvolutionalDictionaryLearner(k, windowSize, algorithm='samples', avoidSingletons=True)
        D = cdl.train(inputTrain)
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
        
        if level < nbLevels - 1:
            # NOTE: for all levels but the last one, return the coefficients from the last level only, without redistributing the activations to lower levels
            coefficients, residual = hcsc.encode(testSignal, nbNonzeroCoefs=None, toleranceSnr=snr, nbBlocks=100, alpha=0.0, singletonWeight=10.0, returnDistributed=False)
            inputTrain = coefficients[-1].todense()
        else:
            coefficients, residual = hcsc.encode(testSignal, nbNonzeroCoefs=None, toleranceSnr=snr, nbBlocks=100, alpha=0.0, singletonWeight=10.0, returnDistributed=True)
    
    # Analyze coefficients
    logger.info('Analyzing final coefficients...')
    nbTotalCoefficients = np.sum([c.nnz for c in coefficients])
    logger.info('Total number of coefficients: %d' % (nbTotalCoefficients))
    for level, c in enumerate(coefficients):
        logger.info('Number of coefficients at level %d: %d (%4.1f%%)' % (level,c.nnz, float(c.nnz)/nbTotalCoefficients*100.0))
    
    multilevelDict.visualize()
    
    logger.info('All done.')
    plt.show()
    