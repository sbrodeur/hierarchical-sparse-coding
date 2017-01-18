
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
import numpy as np
import logging

logger = logging.getLogger(__name__)

def profileFunction(func):
    
    import cProfile, pstats, StringIO
    pr = cProfile.Profile()
    pr.enable()
    
    ret = func()
    
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    logger.info(s.getvalue())
    
    return ret

def findGridSize(l):
    assert l > 0
    n = int(np.ceil(np.sqrt(l)))
    found = False
    for m in range(n+1):
        if m*n >= l:
            found = True
            break
    if not found:
        raise Exception('Unable to find grid size for length %d' % (l))
    return m,n

def normalize(X, axis=None):
    assert X.ndim >= 1
    if axis is None and X.ndim > 1:
        axis = tuple([a for a in range(1,X.ndim)])
    l2norms = np.sqrt(np.sum(np.square(X), axis=axis, keepdims=True))
    # Avoid division by zero
    l2norms = np.where(l2norms > 0.0, l2norms, np.ones_like(l2norms))
    return X / l2norms

def overlapAdd(signal, element, t, copy=False):
    
    if copy:
        signal = np.copy(signal)
    
    # Additive overlap to the signal, taking into account boundaries
    width = element.shape[0]
    startIdx = 0
    endIdx = width
    if np.mod(width, 2) == 0:
        # Even scale
        if t-(width/2-1) < 0:
            startIdx = -(t-(width/2-1))
        if t+width/2+1 > signal.shape[0]:
            endIdx = width - (t+width/2+1 - signal.shape[0])
        
        if endIdx - startIdx > 0:
            signal[max(0,t-width/2+1):min(signal.shape[0],t+width/2+1)] += element[startIdx:endIdx]
    else:
        # Odd scale
        if t-(width/2) < 0:
            startIdx = -(t-(width/2))
        if t+width/2+1 > signal.shape[0]:
            endIdx = width - (t+width/2+1 - signal.shape[0])
            
        if endIdx - startIdx > 0:
            signal[max(0,t-width/2):min(signal.shape[0],t+width/2+1)] += element[startIdx:endIdx]

    return signal

def overlapReplace(signal, element, t, copy=False):
    
    if copy:
        signal = np.copy(signal)
    
    # Overlap to the signal, taking into account boundaries
    width = element.shape[0]
    startIdx = 0
    endIdx = width
    if np.mod(width, 2) == 0:
        # Even scale
        if t-(width/2-1) < 0:
            startIdx = -(t-(width/2-1))
        if t+width/2+1 > signal.shape[0]:
            endIdx = width - (t+width/2+1 - signal.shape[0])
        
        if endIdx - startIdx > 0:
            signal[max(0,t-width/2+1):min(signal.shape[0],t+width/2+1)] =  element[startIdx:endIdx]
    else:
        # Odd scale
        if t-(width/2) < 0:
            startIdx = -(t-(width/2))
        if t+width/2+1 > signal.shape[0]:
            endIdx = width - (t+width/2+1 - signal.shape[0])
            
        if endIdx - startIdx > 0:
            signal[max(0,t-width/2):min(signal.shape[0],t+width/2+1)] = element[startIdx:endIdx]

    return signal