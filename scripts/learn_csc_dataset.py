
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

from hsc.dataset import MultilevelDictionary
from hsc.modeling import ConvolutionalMatchingPursuit, ConvolutionalSparseCoder, ConvolutionalDictionaryLearner
from hsc.analysis import calculateBitForDatatype
from hsc.utils import findGridSize

logger = logging.getLogger(__name__)

def visualize(D, level=0, maxCounts=None):
    
    fig = plt.figure(figsize=(8,8), facecolor='white', frameon=True)
    fig.canvas.set_window_title('Level %d dictionary' % (level))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        hspace=0.01, wspace=0.01)
    
    count = D.shape[0]
    if maxCounts is not None:
        if isinstance(maxCounts, collections.Iterable):
            count = min(count, int(maxCounts[level]))
        else:
            count = min(count, int(maxCounts))
        
    m,n = findGridSize(count)
    
    dict = D[:count]
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
    return fig
 
if __name__ == "__main__":

    # Fix seed for random number generation for reproducible results
    np.random.seed(42)

    logging.basicConfig(level=logging.INFO)
    cdir = os.path.dirname(os.path.realpath(__file__))
    
    # Load reference multilevel dictionary from file
    filePath = os.path.join(cdir, 'multilevel-dict.pkl')
    multilevelDict = MultilevelDictionary.restore(filePath)
    
    # Load the referencesignal from file
    filePath = os.path.join(cdir, 'dataset.npz')
    data = np.load(filePath)
    signal = data['signal']
    events = data['events']
    
    logger.info('Number of samples in dataset: %d' % (len(signal)))
    logger.info('Number of events in dataset: %d' % (len(events)))
    
    # Learning the dictionary
    logger.info('Learning dictionary...')
    cdl = ConvolutionalDictionaryLearner(k=64, windowSize=32, algorithm='kmean')
    D = cdl.train(signal, nbRandomWindows=4096, maxIterations=100, tolerance=0.0, resetMethod='random_samples')
    
    # Visualize dictionary and save to disk as images
    fig = visualize(D, maxCounts=16)
    fig.savefig(os.path.join(cdir, 'learned-dict-l0.eps'), format='eps', dpi=1200)

#     import cProfile, pstats, StringIO
#     pr = cProfile.Profile()
#     pr.enable()

    logger.info('Encoding signal...')
    cmp = ConvolutionalMatchingPursuit()
    csc = ConvolutionalSparseCoder(D, approximator=cmp)
    coefficients, residual = csc.encode(signal, nbNonzeroCoefs=None, toleranceSnr=25.0, nbBlocks=1000, alpha=0.5)

#     pr.disable()
#     s = StringIO.StringIO()
#     sortby = 'cumulative'
#     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#     ps.print_stats()
#     print s.getvalue()
    
    #plt.show()
    
    pidx_bits = np.ceil(np.log(D.shape[0])/np.log(2))
    sidx_bits = 0
    c_bits = calculateBitForDatatype(dtype=signal.dtype)
    bits = sidx_bits + pidx_bits + c_bits
    bitsEncoded = bits * coefficients.nnz
    bitsRaw = c_bits * len(signal)
    logger.info('Bitrate before encoding: %f bit/sample' % (float(bitsRaw) / len(signal)))
    logger.info('Bitrate after encoding: %f bit/sample' % (float(bitsEncoded) / len(signal)))
    logger.info('Compression ratio (raw/encoded): %f' % (float(bitsRaw) / bitsEncoded))
    
    logger.info('All done.')
    