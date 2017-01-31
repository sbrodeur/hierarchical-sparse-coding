
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
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hsc.dataset import MultilevelDictionary, MultilevelDictionaryGenerator, SignalGenerator
from hsc.analysis import calculateMultilevelInformationRates, calculateBitForDatatype

logger = logging.getLogger(__name__)


class ToySignalGenerator(SignalGenerator):

    def generateEvents(self, nbSamples):
        
        events = []
        scales = self.multilevelDict.scales
        remainingLength = nbSamples
        nextMinTime = 0.0
        while nextMinTime < nbSamples:
            
            # Sample across levels
            level = np.random.randint(low=0, high=self.multilevelDict.getNbLevels(), size=1)
            if np.mod(scales[level], 2) == 0:
                # Even scale
                time = nextMinTime + scales[level]/2-1 + np.random.randint(low=0, high=scales[level]/2)
            else:
                # Odd scale
                time = nextMinTime + scales[level]/2 + np.random.randint(low=0, high=scales[level]/2 + 1)
            nextMinTime = time + scales[level]/2
        
            # Sample basis at selected level
            index = np.random.randint(low=0, high=self.multilevelDict.counts[level], size=1)
        
            # Sample coefficients
            coefficient = np.random.uniform(low=0.25, high=4.0, size=1).astype(np.float32)
            
            event = (time, level, index, coefficient)
            events.append(event)
                
        # Sort events by time (increasing)
        events = sorted(events, key=lambda x: x[0])
                
        # Convert to mixed-type numpy array
        events = np.array(events, dtype=('int32,int32,int32,float32'))
                
        return events

if __name__ == "__main__":

    # Fix seed for random number generation for reproducible results
    # !!! DO NOT CHANGE THE SEED IF YOU WISH TO GENERATE THE REFERENCE DATASET !!!
    np.random.seed(42)
    
    # NOTE: set so that any numerical error will raise an exception
    np.seterr(all='raise')
    
    logging.basicConfig(level=logging.INFO)
    cdir = os.path.dirname(os.path.realpath(__file__))
    
    # Generate multi-level dictionary
    logger.info('Generating new multi-level dictionary...')
    scales = [16, 64, 128, 256]
    counts = [4, 8, 16, 32]
    decompositionSizes = [3, 3, 2, 2]
    logger.info('Scales defined for levels: %s' % (str(scales)))
    logger.info('Counts defined for levels: %s' % (str(counts)))
    logger.info('Decomposition sizes defined for levels: %s' % (str(decompositionSizes)))
    
    mlgd = MultilevelDictionaryGenerator()
    multilevelDict = mlgd.generate(scales, counts, decompositionSize=decompositionSizes,
                                   positionSampling='no-overlap', weightSampling='random', multilevelDecomposition=False,
                                   maxNbPatternsConsecutiveRejected=100, nonNegativity=False)
    
    # Save multi-level dictionary to disk, as the reference
    filePath = os.path.join(cdir, 'multilevel-dict-toy.pkl')
    logger.info('Saving dictionary to file: %s' % (filePath))
    multilevelDict.save(filePath)
    
    # Visualize dictionary and save to disk as images
    logger.info('Generating dictionary visualizations...')
    figs = multilevelDict.visualize(maxCounts=16)
    for l,fig in enumerate(figs):
        fig.savefig(os.path.join(cdir, 'dict-l%d-toy.eps' % (l)), format='eps', dpi=1200)
    
    # Generate training and testing datasets
    for datasetName, nbSamples in [('train', int(1e7)), ('test', int(1e6))]:
    
        # Generate events and signal using the multi-level dictionary
        logger.info('Generating events and raw temporal signal for dataset %s...' % (datasetName))
        generator = ToySignalGenerator(multilevelDict, rates=[None] * multilevelDict.getNbLevels())
        events = generator.generateEvents(nbSamples)
        signal = generator.generateSignalFromEvents(events, nbSamples=nbSamples)
        logger.info('Number of generated events: %d , in %d samples' % (len(events), len(signal)))
        
        # Estimate rates from event
        rates = []
        eventLevels = np.array([event[1] for event in events], dtype=np.int)
        for level in range(multilevelDict.getNbLevels()):
            nbEvents = np.count_nonzero(eventLevels == level)
            rate = float(nbEvents) / (nbSamples * multilevelDict.counts[level])
            rates.append(rate)
        rates = np.array(rates, dtype=np.float32)
        logger.info('Estimated rates from events: %s' % (str(rates)))
        
        # Save signal to disk, as the reference
        np.savez_compressed('dataset-%s-toy.npz' % (datasetName), signal=signal, events=events, rates=rates)
    
    # Visualize the beginning of the signal and save image to disk
    logger.info('Generating figures for visualization...')
    shortSignal = signal[:min(10000, len(signal))]
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
    fig.savefig(os.path.join(cdir, 'signal-toy.eps'), format='eps', dpi=1200)
    
    logger.info('All done.')
    plt.show()
    