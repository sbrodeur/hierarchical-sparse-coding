
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
import scipy
import scipy.sparse
import matplotlib
import matplotlib.pyplot as plt

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
    
def calculateEmpiricalInformationRates(sequence, coefficients, multilevelDict):

    # Raw data analysis
    rawBits = calculateBitForDatatype(dtype=sequence.dtype) * len(sequence)
    rawInfoRate = float(rawBits) / len(sequence)

    # Sparse coefficient analysis
    bits = calculateBitForLevels(multilevelDict, dtype=coefficients[0].dtype)
    sparseBits = 0
    for level in range(len(coefficients)):
        if scipy.sparse.issparse(coefficients[level]):
            nbEvents = coefficients[level].nnz
        else:
            nbEvents = coefficients[level].shape[0]
        sparseBits += nbEvents * bits[level]
    sparseInfoRate = float(sparseBits) / len(sequence)
    
    return rawInfoRate, sparseInfoRate
    
def calculateDistributionRatios(coefficients):
    # Coefficient distribution across levels
    nbTotalCoefficients = np.sum([c.nnz for c in coefficients])
    coefficientsDistribution = np.array([float(c.nnz)/nbTotalCoefficients for c in coefficients])
    return coefficientsDistribution

def visualizeDistributionRatios(weights, distributions):
    assert len(weights) > 0
    assert len(distributions) == len(weights)

    cumulatives = np.cumsum(distributions, axis=1)

    fig = plt.figure(figsize=(8,4), facecolor='white', frameon=True)
    fig.canvas.set_window_title('Effect of weighting on the coefficient distribution')
    fig.subplots_adjust(left=0.1, right=1.0, bottom=0.15, top=0.95,
                        hspace=0.01, wspace=0.01)
    ax = fig.add_subplot(111)
    
    bar_width = 0.75
    bar_left = [i+1 for i in range(len(weights))] 
    tick_pos = [i+(bar_width/2) for i in bar_left] 
    
    # NOTE: reverse order so that the legend show the level in the same order as the bars
    bars = []
    colors = ['#606060', '#B8B8B8', '#E0E0E0', '#F0F0F0', '#FFFFFF']
    for level in range(len(distributions[0]))[::-1]:
        if level > 0:
            bottom = cumulatives.T[level-1]
        else:
            bottom = None
        bar = ax.bar(bar_left, distributions.T[level], bar_width, label='Level %d' % (level), bottom=bottom, color=colors[level])
        bars.append(bar)
    
    ax.set_ylabel('Distribution ratio')
    ax.set_xlabel('Singleton weighting')
    ax.set_xticks(tick_pos) 
    ax.set_xticklabels(['%4.2f' % (w) for w in weights])
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax.set_xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])
    ax.set_ylim([0.0, 1.1])
    
    # Add legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.83, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    return fig
    
def visualizeInformationRates(weights, sparseInfoRates, showAsBars=False):
    assert len(weights) > 0
    assert len(sparseInfoRates) == len(weights)

    fig = plt.figure(figsize=(8,4), facecolor='white', frameon=True)
    fig.canvas.set_window_title('Effect of weighting on the information rates')
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.95,
                        hspace=0.01, wspace=0.01)
    ax = fig.add_subplot(111)
    
    if showAsBars:
        bar_width = 0.75
        bar_left = [i+1 for i in range(len(weights))] 
        tick_pos = [i+(bar_width/2) for i in bar_left] 
        bar = ax.bar(bar_left, sparseInfoRates, bar_width, color='#606060')
        
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(['%4.2f' % (w) for w in weights])
        ax.set_xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])
    else:
        ax.plot(weights, sparseInfoRates, color='k', marker='.', markersize=10)
        ax.set_xlim([0.0, 1.1 * np.max(weights)])
    
    ax.set_ylim([0.0, 1.1 * np.max(sparseInfoRates)])
    ax.set_ylabel('Information rate [bit/sample]')
    ax.set_xlabel('Singleton weighting')
    
    return fig
    
def visualizeEnergies(weights, energies, showAsBars=False, signalEnergy=None):
    assert len(weights) > 0
    assert len(energies) == len(weights)

    fig = plt.figure(figsize=(8,4), facecolor='white', frameon=True)
    fig.canvas.set_window_title('Effect of weighting on the coefficient energies')
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.95,
                        hspace=0.01, wspace=0.01)
    ax = fig.add_subplot(111)
    
    if showAsBars:
        bar_width = 0.75
        bar_left = [i+1 for i in range(len(weights))] 
        tick_pos = [i+(bar_width/2) for i in bar_left] 
        bar = ax.bar(bar_left, energies, bar_width, color='#606060')
        
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(['%4.2f' % (w) for w in weights])
        ax.set_xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])
    else:
        ax.plot(weights, energies, color='k', marker='.', markersize=10)
        ax.set_xlim([0.0, 1.1 * np.max(weights)])
    
    if signalEnergy is not None:
        ax.axhline(y=signalEnergy, linewidth=2, color='r')
    
    ax.set_ylim([0.0, 1.1 * np.max(energies)])
    ax.set_ylabel('Energy')
    ax.set_xlabel('Singleton weighting')
    
    return fig

def visualizeInformationRatesOptimality(scales, sparseInfoRates, optimalScales=None, optimalInfoRates=None):
    
    # Visualize information rate distribution across levels
    fig = plt.figure(figsize=(8,8), facecolor='white', frameon=True)
    ax = fig.add_subplot(111)
    
    ax.plot(scales, sparseInfoRates, '-k', label='Empirical')
    
    if optimalScales is not None and optimalInfoRates is not None:
        # Lower bound
        ax.plot(optimalScales, optimalInfoRates, '--k', label='Lower bound')
        
    ax.set_title('Information rate distribution')
    ax.set_xlabel('Maximum scale')
    ax.set_ylabel('Information rate [bit/sample]')
    ax.legend()
    
    return fig

    