
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

from hsc.dataset import MultilevelDictionary, MultilevelDictionaryGenerator

logger = logging.getLogger(__name__)

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
    scales = [32, 64]
    counts = [4, 1]
    decompositionSizes = 4
    logger.info('Scales defined for levels: %s' % (str(scales)))
    logger.info('Counts defined for levels: %s' % (str(counts)))
    logger.info('Decomposition sizes defined for levels: %s' % (str(decompositionSizes)))
    
    mlgd = MultilevelDictionaryGenerator()
    multilevelDict = mlgd.generate(scales, counts, decompositionSize=decompositionSizes, 
                                   positionSampling='random', multilevelDecomposition=True, 
                                   maxNbPatternsConsecutiveRejected=100, nonNegativity=False)
    
    # Visualize dictionary and save to disk as images
    logger.info('Generating dictionary visualizations...')
    figs = multilevelDict.visualize(shuffle=False, annotate=True)
    for l,fig in enumerate(figs):
        fig.savefig(os.path.join(cdir, 'dict-decomp-l%d.eps' % (l)), format='eps', dpi=1200)
    
    levels, fIdxs, positions, coefficients = multilevelDict.decompositions[0][0]
    for level, fIdx, position, coefficient in zip(levels, fIdxs, positions, coefficients):
        logger.info('Sub-element at level %d, filter index %d, position %d with coefficient %f' % (level, fIdx, position, coefficient))
    
    logger.info('All done.')
    plt.show()
    