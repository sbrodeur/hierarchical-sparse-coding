
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

import collections
import numpy as np

def calculateBitForDatatype(dtype):
    if np.issubdtype(dtype, np.float):
        c_bits = 1 + np.finfo(dtype).iexp + np.finfo(dtype).nmant # sign + exponent + fraction bits
    elif np.issubdtype(dtype, np.int):
        c_bits = np.iinfo(dtype).bits # integer bits (signed or unsigned)
    else:
        raise Exception('Unsupported datatype: %s' % (str(dtype)))
    return c_bits

def calculateBitForLevels(multilevelDict, dtype=np.float32):
    
    # Find the number of bit to describe a pattern at each scale (depends on the counts)
    pidx_bits = np.ceil(np.log(multilevelDict.counts)/np.log(2))
    sidx_bits = np.ceil(np.log(len(multilevelDict.scales))/np.log(2))
    
    c_bits = calculateBitForDatatype(dtype)
        
    bits = sidx_bits + pidx_bits + c_bits
    
    return bits

def calculateInformationRate(multilevelDict, rates, dtype=np.float32):
    assert len(rates) == multilevelDict.getNbLevels()
    
    bits = calculateBitForLevels(multilevelDict, dtype)
    
    avgInfoRate = 0.0
    for level in range(multilevelDict.getNbLevels()):
        # Compute the average information rate, in bit/sample.
        # Each scale is considered a homogeneous Poisson process at a given rate.
        avgInfoRate += np.sum(rates[level] * bits[level])
    
    return avgInfoRate

def calculateMultilevelInformationRates(multilevelDict, rates, dtype=np.float32):
    assert len(rates) == multilevelDict.getNbLevels()
    
    if not isinstance(rates[0], collections.Iterable):
        rates = [rates[level] * np.ones(multilevelDict.counts[level]) for level in range(multilevelDict.getNbLevels())]
    
    # Loop over for all levels, starting from the last
    avgInfoRates = []
    for level in reversed(range(multilevelDict.getNbLevels())):
        
        # Calculate information rate
        avgInfoRate = calculateInformationRate(multilevelDict, rates)
        avgInfoRates.append(avgInfoRate)
        
        if level > 0:
            # Redistribute rates at current level to the previous levels, based on the decomposition scheme.
            # Loop over all elements at current level
            decompositions = multilevelDict.decompositions[level-1]
            for n, rate, [selectedLevels, selectedIndices, _, _] in zip(range(len(decompositions)), rates[level], decompositions):
                # Loop over all sub-elements at previous levels
                for l,i in zip(selectedLevels, selectedIndices):
                    rates[l][i] += rate
                # Remove the rate contribution of the element at current level
                rates[level][n] = 0.0
            
            assert np.allclose(np.sum(rates[level]), 0.0)
    
    # Convert to numpy array and reverse
    avgInfoRates = np.array(avgInfoRates)[::-1]
    return avgInfoRates
    