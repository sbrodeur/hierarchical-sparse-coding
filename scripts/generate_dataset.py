
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
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from hsc.dataset import MultilevelDictionary, SignalGenerator
from hsc.analysis import calculateMultilevelInformationRates, calculateBitForDatatype

logger = logging.getLogger(__name__)

def generateScales(mode='linear'):
    if mode == 'linear':
        scales = np.arange(1, 15, 2).astype(np.int)
    elif mode == 'exponential':
        scales = 2**np.arange(5, 10, dtype=np.int)
    elif mode == 'prime':
        scales = np.array([1, 2, 3, 5, 7, 11, 13, 17], dtype=np.int)
    else:
        raise Exception('Unsupported mode: %s' % (mode))
    return scales

def generateCountDistribution(scales, mode='constant'):
    scales = np.array(scales)
    if mode == 'linear':
        # Linear scaling of pattern counts
        counts = np.linspace(16, 256, len(scales), dtype=np.int)
    elif mode == 'constant':
        counts = 2 * np.ones_like(scales).astype(dtype=np.int)
    elif mode == 'exponential':
        counts = 2 ** (np.arange(1, len(scales)+1, dtype=np.int))
    else:
        raise Exception('Unsupported mode: %s' % (mode))
    return counts

def generateRateDistribution(scales, mode='constant'):
    
    scales = np.array(scales)
    if mode == 'constant':
        r = 0.00002 * np.ones_like(scales)
    elif mode == 'linear':
        r = 0.00005 * scales / np.max(scales)
    elif mode == 'gaussian':
        # Normal distribution for pattern rates across scales.
        # Set mean as the median scale, and standard deviation as 
        mu = 2.25 * np.mean(scales)
        sigma = 0.5 * mu
        factor = 0.01
        r = 1.0/(np.sqrt(2.0 * sigma**2 * np.pi)) * np.exp(-(scales - mu)**2 / (2.0 * sigma**2))
        r /= np.sum(r)
        r *= factor
    else:
        raise Exception('Unsupported mode: %s' % (mode))
    
    # NOTE: rate is in the inverval [0,1], in event/sample
    return r

if __name__ == "__main__":

    # Fix seed for random number generation for reproducible results
    # !!! DO NOT CHANGE THE SEED IF YOU WISH TO GENERATE THE REFERENCE DATASET !!!
    np.random.seed(42)

    logging.basicConfig(level=logging.INFO)
    cdir = os.path.dirname(os.path.realpath(__file__))
    
    # Generate multi-level dictionary
    logger.info('Generating new multi-level dictionary...')
    scales = generateScales(mode='exponential')
    counts = generateCountDistribution(scales, mode='linear')
    decompositionSizes = np.linspace(4, 8, len(scales), dtype=np.int)
    logger.info('Scales defined for levels: %s' % (str(scales)))
    logger.info('Counts defined for levels: %s' % (str(counts)))
    logger.info('Decomposition sizes defined for levels: %s' % (str(decompositionSizes)))
    multilevelDict = MultilevelDictionary(scales, counts,
                                          decompositionSizes, maxNbPatternsConsecutiveRejected=100)
    # Visualize dictionary and save to disk as images
    figs = multilevelDict.visualize(maxCounts=9)
    for l,fig in enumerate(figs):
        fig.savefig(os.path.join(cdir, 'dict-l%d.eps' % (l)), format='eps', dpi=1200)
    
    # Save multi-level dictionary to disk, as the reference
    filePath = os.path.join(cdir, 'multilevel-dict.pkl')
    multilevelDict.save(filePath)
    
    # Find the optimal rates to achieve about 50% of the raw bitrate when reducing to the first level
    logger.info('Estimating optimal rates...')
    minimumCompressionRatio = 0.50
    dtype = np.float32
    rates = generateRateDistribution(multilevelDict.scales, mode='constant')
    c_bits = calculateBitForDatatype(dtype)
    factors = np.linspace(1e-6, 1.0, num=1000)[::-1]
    factorIdx = 0
    while True:
        scaledRates = np.copy(rates) * factors[factorIdx]
        avgInfoRates = calculateMultilevelInformationRates(multilevelDict, scaledRates, dtype=dtype)
        if avgInfoRates[0] <= c_bits * minimumCompressionRatio:
            # Valid rates found
            break
        else:
            factorIdx += 1
            if factorIdx >= len(factors):
                raise Exception("Unable to find the optimal rates: initial rates are too high")
            logger.debug('Rates are too high (bitrate of %f bit/sample at first level): scaling so that maximum rate across levels is %f' % (avgInfoRates[0], np.max(rates)))
    rates = scaledRates
    logger.info('Optimal rate scale found: %f (for bitrate of %f bit/sample at first level)' % (np.max(rates), avgInfoRates[0]))
    
    # Generate events and signal using the multi-level dictionary
    logger.info('Generating events and raw temporal signal...')
    nbSamples = int(1e7)
    generator = SignalGenerator(multilevelDict, rates)
    events = generator.generateEvents(nbSamples=nbSamples)
    signal = generator.generateSignalFromEvents(events, nbSamples=nbSamples)
    logger.info('Number of generated events: %d' % (len(events)))
    
    # Visualize the beginning of the signal and save image to disk
    logger.info('Generating figures for visualization...')
    shortSignal = signal[:min(100000, len(signal))]
    fig = plt.figure(figsize=(8,4), facecolor='white', frameon=True)
    fig.canvas.set_window_title('Generated signal')
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.95,
                        hspace=0.01, wspace=0.01)
    ax = fig.add_subplot(111)
    n = np.arange(len(shortSignal))
    ax.plot(n, shortSignal, color='k')
    ax.set_xlim(0, len(shortSignal))
    r = np.max(np.abs(shortSignal))
    ax.set_ylim(-r, r)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Amplitude')
    fig.savefig(os.path.join(cdir, 'signal.eps'), format='eps', dpi=1200)
    
    # Save signal to disk, as the reference
    np.savez_compressed('dataset.npz', signal=signal, events=events, rates=rates)
    
    plt.show()
    
    logger.info('All done.')
    