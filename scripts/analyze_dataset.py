
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

from hsc.dataset import MultilevelDictionary
from hsc.analysis import calculateMultilevelInformationRates

logger = logging.getLogger(__name__)

def showCountRatePlot(multilevelDict, rates):
    fig = plt.figure(figsize=(12,6),facecolor='white', frameon=True)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.10, top=0.95,
                        hspace=0.01, wspace=0.3)
    
    ax = fig.add_subplot(121)
    markerline, _, _ = ax.stem(multilevelDict.scales, multilevelDict.counts, 'k')
    plt.setp(markerline, 'markerfacecolor', 'k')
    ax.set_title('Count distribution')
    ax.set_xlabel('Scale')
    ax.set_ylabel('Count')
    
    ax = fig.add_subplot(122)
    markerline, _, _ = ax.stem(multilevelDict.scales, rates, 'k')
    plt.setp(markerline, 'markerfacecolor', 'k')
    ax.set_title('Rate distribution')
    ax.set_xlabel('Scale')
    ax.set_ylabel('Rate [event/sample]')
    
    return fig

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
    
    # Load the referencesignal from file
    filePath = os.path.join(cdir, 'dataset.npz')
    data = np.load(filePath)
    rates = data['rates']
    
    # Calculate raw input information rate
    dtype = np.float32
    if np.issubdtype(dtype, np.float):
        inputInfoRate = 1 + np.finfo(dtype).iexp + np.finfo(dtype).nmant # sign + exponent + fraction bits
    elif np.issubdtype(dtype, np.int):
        inputInfoRate = np.iinfo(dtype).bits # integer bits
    else:
        raise Exception('Unsupported datatype: %s' % (str(dtype)))
    avgInfoRates = calculateMultilevelInformationRates(multilevelDict, rates, dtype=dtype)
    compRatioRates = inputInfoRate / avgInfoRates[-1]
    
    # Print statistics about information rate 
    logger.info('Input information rate: %f bit/sample' % (inputInfoRate))
    logger.info('Maximum average compressed information rate: %f bit/sample' % (avgInfoRates[-1]))
    logger.info('Maximum average compression ratio: %f' % (compRatioRates))
    
    # Visualize information rate distribution across levels
    fig = plt.figure(figsize=(8,8), facecolor='white', frameon=True)
    ax = fig.add_subplot(111)
    ax.plot(multilevelDict.scales, avgInfoRates, '-k', label='Lower bound')
    ax.set_title('Information rate distribution')
    ax.set_xlabel('Maximum scale')
    ax.set_ylabel('Information rate [bit/sample]')
    ax.legend()
    fig.savefig(os.path.join(cdir, 'info-rate-levels.eps'), format='eps', dpi=1200)
    
    fig = showCountRatePlot(multilevelDict, rates)
    fig.savefig(os.path.join(cdir, 'count-rate.eps'), format='eps', dpi=1200)
    
    plt.show()
    