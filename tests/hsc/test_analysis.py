
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
import unittest
import numpy as np

from hsc.dataset import MultilevelDictionary
from hsc.analysis import calculateBitForLevels, calculateInformationRate, calculateMultilevelInformationRates

class TestFunctions(unittest.TestCase):

    def test_calculateBitForLevels(self):
        multilevelDict = MultilevelDictionary(scales=[32,64], counts=[8, 16])
        bits = calculateBitForLevels(multilevelDict, dtype=np.float32)
        self.assertTrue(len(bits) == multilevelDict.getNbLevels())
        # First level: 1bit scale + 3 bit index + 32bit value = 36 bits
        # Second level: 1bit scale + 4 bit index + 32bit value = 37 bits
        self.assertTrue(np.allclose(bits, [36, 37]))
        
        bits = calculateBitForLevels(multilevelDict, dtype=np.int16)
        self.assertTrue(len(bits) == multilevelDict.getNbLevels())
        # First level: 1bit scale + 3 bit index + 16bit value = 20 bits
        # Second level: 1bit scale + 4 bit index + 16bit value = 21 bits
        self.assertTrue(np.allclose(bits, [20, 21]))
        
    def test_calculateInformationRate(self):
        rates = [0.01]
        multilevelDict = MultilevelDictionary(scales=[32], counts=[8])
        avgInfoRate = calculateInformationRate(multilevelDict, rates)
        self.assertTrue(np.isscalar(avgInfoRate))
        # First level: 0bit scale + 3 bit index + 32bit value = 35 bits
        self.assertTrue(np.allclose(avgInfoRate, 35 * 0.01))
        
        rates = [0.01, 0.02]
        multilevelDict = MultilevelDictionary(scales=[32,64], counts=[8, 16])
        avgInfoRate = calculateInformationRate(multilevelDict, rates)
        self.assertTrue(np.isscalar(avgInfoRate))
        # First level: 1bit scale + 3 bit index + 32bit value = 36 bits
        # Second level: 1bit scale + 4 bit index + 32bit value = 37 bits
        self.assertTrue(np.allclose(avgInfoRate, 36 * 0.01 + 37 * 0.02))
        
    def test_calculateMultilevelInformationRates(self):
        
        rates = [0.01, 0.02]
        multilevelDict = MultilevelDictionary(scales=[32,64], counts=[8, 16], decompositionSize=3)
        avgInfoRates = calculateMultilevelInformationRates(multilevelDict, rates, dtype=np.float32)
        # First level: 1bit scale + 3 bit index + 32bit value = 36 bits
        # Second level: 1bit scale + 4 bit index + 32bit value = 37 bits
        self.assertTrue(np.allclose(avgInfoRates, [36 * 0.01 * 8 + 36 * 0.02 * 3 * 16, 
                                                   36 * 0.01 * 8 + 37 * 0.02 * 16]))
        
        rates = [0.0001, 0.0002, 0.0004, 0.0004]
        multilevelDict = MultilevelDictionary(scales=[32,64,128,256], counts=[8, 16, 32, 64], decompositionSize=4)
        avgInfoRates = calculateMultilevelInformationRates(multilevelDict, rates, dtype=np.float32)
        self.assertTrue(np.array_equal(np.argsort(avgInfoRates), np.arange(len(avgInfoRates))[::-1]))
        
if __name__ == '__main__':
    unittest.main()
