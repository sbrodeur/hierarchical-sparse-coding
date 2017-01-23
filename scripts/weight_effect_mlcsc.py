
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
import argparse
import ctypes
import logging
import cPickle as pickle
import collections
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from multiprocessing import Pool

from hsc.dataset import MultilevelDictionary, scalesToWindowSizes, addSingletonBases
from hsc.modeling import HierarchicalConvolutionalMatchingPursuit, HierarchicalConvolutionalSparseCoder, ConvolutionalDictionaryLearner
from hsc.analysis import calculateEmpiricalInformationRates, calculateDistributionRatios, visualizeDistributionRatios, visualizeInformationRates, visualizeEnergies
from hsc.utils import findGridSize, profileFunction

logger = logging.getLogger(__name__)

def learnMultilevelDictionary(trainSignal, testSignal, weight):

    # Learn multilevel dictionary on the training data
    dictionaries = []
    input = trainSignal
    scales = [32, 64, 128, 256, 512]
    counts = [16, 32, 64, 128, 256]
    snr = 40.0
    nbLevels = len(counts)
    widths = scalesToWindowSizes(scales)
    for level, k, windowSize in zip(range(nbLevels), counts, widths):
        
        # Learning dictionary
        cdl = ConvolutionalDictionaryLearner(k, windowSize, algorithm='samples', avoidSingletons=True)
        D = cdl.train(input)
        assert D.shape[0] == k
        dictionaries.append(D)
    
        # Encoding signal
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
            coefficients, residual = hcsc.encode(trainSignal, toleranceSnr=snr, nbBlocks=100, alpha=0.0, singletonWeight=weight, returnDistributed=False)
        elif level < nbLevels - 1:
            coefficients = hcsc.encodeFromLevel(coefficients, toleranceSnr=snr, nbBlocks=100, alpha=0.0, singletonWeight=weight, returnDistributed=False)
        input = coefficients[-1].todense()

    # Analyze encoding on the test signal
    hcmp = HierarchicalConvolutionalMatchingPursuit()
    hcsc = HierarchicalConvolutionalSparseCoder(multilevelDict, approximator=hcmp)
    coefficients, residual = hcsc.encode(testSignal, toleranceSnr=snr, nbBlocks=100, alpha=0.0, singletonWeight=weight, returnDistributed=True)
    
    return weight, multilevelDict, coefficients

if __name__ == "__main__":

    # Fix seed for random number generation for reproducible results
    np.random.seed(42)

    # NOTE: set so that any numerical error will raise an exception
    np.seterr(all='raise')

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--overwrite-results", help="force to overwrite results",
                        action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cdir = os.path.dirname(os.path.realpath(__file__))
    outputResultPath = os.path.join(cdir, 'mlcsc-weight-effect')
    if not os.path.exists(outputResultPath):
        os.makedirs(outputResultPath)
    resultsFilePath = os.path.join(outputResultPath, 'results.pkl')
    
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
    
    if not os.path.exists(resultsFilePath) or args.overwrite_results:
        
        # Values of weights to evaluate
        weights = np.array([0.5, 0.75, 0.8, 0.9, 2.0])
        def f(w):
            return learnMultilevelDictionary(trainSignal, testSignal, w)
        
        # Use all available cores on the CPU
        p = Pool()
        results = p.map(f, weights)
        
        # Save results to file, for later analysis
        with open(resultsFilePath, 'wb') as file:
            pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Load results from file
    with open(resultsFilePath, 'rb') as file:
        results = pickle.load(file)
    
    # Extract results
    weights = []
    sparseInfoRates = []
    distributions = []
    energies = []
    for weight, multilevelDict, coefficients in results:
        
        weights.append(weight)
        
        # Coefficient distribution across levels
        distribution = calculateDistributionRatios(coefficients)
        distributions.append(distribution)
        
        # Information rate
        rawInfoRate, sparseInfoRate = calculateEmpiricalInformationRates(testSignal, coefficients, multilevelDict)
        sparseInfoRates.append(sparseInfoRate)
        
        # Coefficient energies
        energy = 0.0
        for c in coefficients:
            energy += c.multiply(c).sum()
        energies.append(energy)
        
    weights = np.array(weights)
    sparseInfoRates = np.array(sparseInfoRates)
    distributions = np.array(distributions)
    energies = np.array(energies)
    
    # Information rate analysis
    logger.info('Analysing information rates...')
    fig = visualizeInformationRates(weights, sparseInfoRates, showAsBars=True)
    fig.savefig(os.path.join(outputResultPath, 'information-rate-effect.eps'), format='eps', dpi=1200)
    
    # Distribution analysis
    logger.info('Analysing distribution across levels...')
    fig = visualizeDistributionRatios(weights, distributions)
    fig.savefig(os.path.join(outputResultPath, 'distribution-effect.eps'), format='eps', dpi=1200)
    
    # Energy analysis
    logger.info('Analysing energies...')
    signalEnergy = np.sum(np.square(testSignal))
    fig = visualizeEnergies(weights, energies, showAsBars=True, signalEnergy=signalEnergy)
    fig.savefig(os.path.join(outputResultPath, 'energy-effect.eps'), format='eps', dpi=1200)
    
    logger.info('All done.')
    plt.show()
    