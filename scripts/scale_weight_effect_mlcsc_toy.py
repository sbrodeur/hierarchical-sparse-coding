
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
import copy
import argparse
import fnmatch
import ctypes
import logging
import cPickle as pickle
import collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from multiprocessing import Pool

from hsc.dataset import MultilevelDictionary, scalesToWindowSizes, addSingletonBases
from hsc.modeling import HierarchicalConvolutionalMatchingPursuit, HierarchicalConvolutionalSparseCoder, ConvolutionalDictionaryLearner
from hsc.analysis import calculateMultilevelInformationRates, calculateEmpiricalInformationRates, calculateDistributionRatios, visualizeDistributionRatios, visualizeInformationRates, visualizeEnergies, visualizeInformationRatesOptimality

logger = logging.getLogger(__name__)

def learnMultilevelDictionary(multilevelDictRef, trainSignal, testSignal, weight, resultsDir, overwrite=True):

    resultFilePath = os.path.join(resultsDir, 'results-w%f.pkl' % weight)
    if not os.path.exists(resultFilePath) or overwrite:

        # Add a handler to the root logger to write to file (per process)
        log = logging.getLogger()
        log.setLevel(logging.DEBUG)
        logFilePath = os.path.join(resultsDir, "process_w%4.2f.log" % (weight))
        if os.path.exists(logFilePath):
            os.remove(logFilePath)
        h = logging.FileHandler(logFilePath)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        h.setFormatter(formatter)
        log.addHandler(h)
        logger.info('Log for process will be written to file: %s' % (logFilePath))
    
        scales = [16, 64, 128, 256]
        snrs = [40.0, 120.0, 120.0]
        widths = scalesToWindowSizes(scales)
        coefficientsForScales = []
        for level in range(multilevelDictRef.getNbLevels()):
            
            multilevelDict = multilevelDictRef.upToLevel(level)
            hcmp = HierarchicalConvolutionalMatchingPursuit()
            hcsc = HierarchicalConvolutionalSparseCoder(multilevelDict, approximator=hcmp)
            
            # NOTE: for all levels but the last one, return the coefficients from the last level only, without redistributing the activations to lower levels
            snr = snrs[level]
            if level == 0:
                testCoefficients, _ = hcsc.encode(testSignal, toleranceSnr=snr, nbBlocks=1, alpha=0.0, singletonWeight=weight, returnDistributed=False)
            else:
                testCoefficients = hcsc.encodeFromLevel(testSignal, testCoefficients, toleranceSnr=snr, nbBlocks=1, alpha=0.0, singletonWeight=weight, returnDistributed=False)
                
            coefficientsForScales.append(testCoefficients)
    
        coefficientsForScales = [hcmp.convertToDistributedCoefficients(coefficients) for coefficients in coefficientsForScales]
        assert len(coefficientsForScales) == multilevelDictRef.getNbLevels()
        
        # Save results to file, for later analysis
        results = (weight, multilevelDict, coefficientsForScales)
        with open(resultFilePath, 'wb') as file:
            pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)

def loadResults(resultsDir):
    
    # Get pickle filenames
    matches = []
    for root, dirnames, filenames in os.walk(resultsDir):
        for filename in fnmatch.filter(filenames, 'results-w*.pkl'):
            matches.append(os.path.join(root, filename))

    # Load objects
    results = []
    for match in matches:
        with open(match, 'rb') as file:
            result = pickle.load(file)
            results.append(result)

    # Sort by weight
    results = sorted(results, key=lambda x: x[0])
    return results

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
    outputResultPath = os.path.join(cdir, 'mlcsc-scale-weight-effect-toy')
    if not os.path.exists(outputResultPath):
        os.makedirs(outputResultPath)
    resultsFilePath = os.path.join(outputResultPath, 'results.pkl')
    
    # Load reference multilevel dictionary from file
    filePath = os.path.join(cdir, 'multilevel-dict-toy.pkl')
    multilevelDictRef = MultilevelDictionary.restore(filePath)
    
    # Load the reference training signal from file
    filePath = os.path.join(cdir, 'dataset-train-toy.npz')
    trainData = np.load(filePath)
    trainSignal = trainData['signal']
    trainEvents = trainData['events']
    logger.info('Number of samples in dataset (training): %d' % (len(trainSignal)))
    logger.info('Number of events in dataset (training): %d' % (len(trainEvents)))
    
    # Load the reference testing signal from file
    filePath = os.path.join(cdir, 'dataset-test-toy.npz')
    testData = np.load(filePath)
    testSignal = testData['signal']
    testEvents = testData['events']
    testRates = testData['rates']
    logger.info('Number of samples in dataset (testing): %d' % (len(testSignal)))
    logger.info('Number of events in dataset (testing): %d' % (len(testEvents)))

    # Reduce the size of the training and testing data
    nbSamples = 100000
    trainSignal = trainSignal[:nbSamples]
    testSignal = testSignal[:nbSamples]
    
    # Values of weights to evaluate
    weights = [0.75, 0.80, 1.0, 2.0]
    def f(w):
        return learnMultilevelDictionary(multilevelDictRef, trainSignal, testSignal, w, resultsDir=outputResultPath, overwrite=args.overwrite_results)
    
    # Use all available cores on the CPU
    p = Pool()
    p.map(f, weights)
    #[f(w) for w in weights]
    
    # Load results from files
    results = loadResults(outputResultPath)
    
    # Get optimal information rate across levels
    optimalInfoRates = calculateMultilevelInformationRates(multilevelDictRef, testRates, dtype=testSignal.dtype)
    optimalScales = multilevelDictRef.scales
    
    # Visualize information rate distribution across levels
    fig = plt.figure(figsize=(8,8), facecolor='white', frameon=True)
    ax = fig.add_subplot(111)

    linestyles = ['-','--', '-.', ':']
    assert len(linestyles) >= len(results)
    for i, (weight, multilevelDict, coefficientsForScales) in enumerate(results):
         
        # Get empirical information rate across levels
        sparseInfoRates = []
        for coefficients in coefficientsForScales:
            rawInfoRate, sparseInfoRate = calculateEmpiricalInformationRates(testSignal, coefficients, multilevelDict)
            sparseInfoRates.append(sparseInfoRate)
        sparseInfoRates = np.array(sparseInfoRates)
    
        # Plot for current weight
        ax.plot(multilevelDict.scales, sparseInfoRates, linestyles[i], color='k', label=r'$\beta$ = %4.2f' % (weight))
    
    # Lower bound
    ax.plot(optimalScales, optimalInfoRates, linestyle='-', color='k', linewidth=3, label='Lower bound')
        
    ax.set_title('Information rate distribution')
    ax.set_xlabel('Maximum scale')
    ax.set_ylabel('Information rate [bit/sample]')
    ax.legend()
    
    fig.savefig(os.path.join(outputResultPath, 'optimality.eps'), format='eps', dpi=1200)
    
    weights = []
    distributions = []
    energies = []
    for i, (weight, multilevelDict, coefficientsForScales) in enumerate(results):
        # Consider all levels
        coefficients = coefficientsForScales[-1]
        
        weights.append(weight)
        
        # Coefficient distribution across levels
        distribution = calculateDistributionRatios(coefficients)
        distributions.append(distribution)
        
        # Coefficient energies
        energy = 0.0
        for c in coefficients:
            c = c.tocsr()
            energy += c.multiply(c).sum()
        energies.append(energy)
        
    weights = np.array(weights)
    distributions = np.array(distributions)
    energies = np.array(energies)
    

    for i, (weight, multilevelDict, coefficientsForScales) in enumerate(results):
        print weight
        # Consider all levels
        for level,coefficients in enumerate(coefficientsForScales):
            # Coefficient distribution across levels
            distribution = calculateDistributionRatios(coefficients)
            print level+1
            print distribution, np.sum([coefficients[l].nnz for l in range(level+1)])
    
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
    