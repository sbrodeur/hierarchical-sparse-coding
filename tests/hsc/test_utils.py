
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

from hsc.utils import findGridSize, normalize, overlapAdd, overlapReplace, peek, profileFunction

class TestFunctions(unittest.TestCase):

    def test_profileFunction(self):
        
        def func():
            x = np.random.random(size=int(1e5))
            return np.argmax(x)
            
        y = profileFunction(func)
        self.assertTrue(y is not None)

    def test_findGridSize(self):
        m,n = findGridSize(16)
        self.assertTrue(m == 4 and n == 4)

        m,n = findGridSize(12)
        self.assertTrue(m == 3 and n == 4)
        
        m,n = findGridSize(8)
        self.assertTrue(m == 3 and n == 3)
        
        m,n = findGridSize(3)
        self.assertTrue(m == 2 and n == 2)
        
        m,n = findGridSize(1)
        self.assertTrue(m == 1 and n == 1)

    def test_normalize_1d(self):
        # 1D vector
        x = np.random.random(size=10)
        xn = normalize(x, axis=None)
        self.assertTrue(xn.shape == x.shape)
        self.assertTrue(np.allclose(np.sqrt(np.sum(np.square(xn))), 1.0))
        xn = normalize(x, axis=0)
        self.assertTrue(xn.shape == x.shape)
        self.assertTrue(np.allclose(np.sqrt(np.sum(np.square(xn))), 1.0))

        # 1D vectors
        x = np.random.random(size=(10,16))
        xn = normalize(x, axis=1)
        self.assertTrue(xn.shape == x.shape)
        for i in range(x.shape[0]):
            self.assertTrue(np.allclose(np.sqrt(np.sum(np.square(xn[i]))), 1.0))
    
        # 1D vectors
        x = np.random.random(size=(10,16))
        xn = normalize(x, axis=None)
        self.assertTrue(xn.shape == x.shape)
        for i in range(x.shape[0]):
            self.assertTrue(np.allclose(np.sqrt(np.sum(np.square(xn[i]))), 1.0))
    
    def test_normalize_2d(self):
        
        # 2D matrices
        x = np.random.random(size=(10,16,4))
        xn = normalize(x, axis=(1,2))
        self.assertTrue(xn.shape == x.shape)
        for i in range(x.shape[0]):
            self.assertTrue(np.allclose(np.sqrt(np.sum(np.square(xn[i]))), 1.0))
            
        # 2D matrices
        x = np.random.random(size=(10,16,4))
        xn = normalize(x, axis=None)
        self.assertTrue(xn.shape == x.shape)
        for i in range(x.shape[0]):
            self.assertTrue(np.allclose(np.sqrt(np.sum(np.square(xn[i]))), 1.0))
            
        # 2D matrices
        x = np.random.random(size=(10,16,4))
        x[0,:,:] = 0.0
        xn = normalize(x, axis=(1,2))
        self.assertTrue(xn.shape == x.shape)
        self.assertTrue(np.allclose(np.sqrt(np.sum(np.square(xn[0]))), 0.0))
        for i in range(1, x.shape[0]):
            self.assertTrue(np.allclose(np.sqrt(np.sum(np.square(xn[i]))), 1.0))

    def test_peek_1d(self):
        
        # Odd width
        sequence = np.arange(8)
        s = peek(sequence, width=5, t=0)
        self.assertTrue(np.allclose(s, [0,1,2]))
        s = peek(sequence, width=5, t=4)
        self.assertTrue(np.allclose(s, [2,3,4,5,6]))
        s = peek(sequence, width=5, t=7)
        self.assertTrue(np.allclose(s, [5,6,7]))

        # Even width
        sequence = np.arange(8)
        s = peek(sequence, width=4, t=0)
        self.assertTrue(np.allclose(s, [0,1,2]))
        s = peek(sequence, width=4, t=4)
        self.assertTrue(np.allclose(s, [3,4,5,6]))
        s = peek(sequence, width=4, t=7)
        self.assertTrue(np.allclose(s, [6,7]))

    def test_peek_2d(self):
        
        # Odd width
        sequence = np.arange(16).reshape((8,2))
        s = peek(sequence, width=5, t=0)
        self.assertTrue(np.allclose(s, [[0,1],[2,3],[4,5]]))
        s = peek(sequence, width=5, t=4)
        self.assertTrue(np.allclose(s, [[4,5],[6,7],[8,9],[10,11],[12,13]]))
        s = peek(sequence, width=5, t=7)
        self.assertTrue(np.allclose(s, [[10,11],[12,13],[14,15]]))

        # Even width
        sequence = np.arange(16).reshape((8,2))
        s = peek(sequence, width=4, t=0)
        self.assertTrue(np.allclose(s, [[0,1],[2,3],[4,5]]))
        s = peek(sequence, width=4, t=4)
        self.assertTrue(np.allclose(s, [[6,7],[8,9],[10,11],[12,13]]))
        s = peek(sequence, width=4, t=7)
        self.assertTrue(np.allclose(s, [[12,13],[14,15]]))

    def test_overlapAdd(self):
        sequence = np.zeros(8)
        element = np.arange(1,6)
        s = overlapAdd(sequence, element, t=0, copy=True)
        self.assertTrue(np.allclose(s, [3,4,5,0,0,0,0,0]))
        s = overlapAdd(sequence, element, t=4, copy=True)
        self.assertTrue(np.allclose(s, [0,0,1,2,3,4,5,0]))
        s = overlapAdd(sequence, element, t=7, copy=True)
        self.assertTrue(np.allclose(s, [0,0,0,0,0,1,2,3]))
        
        sequence = np.zeros(8)
        element = np.arange(1,5)
        s = overlapAdd(sequence, element, t=0, copy=True)
        self.assertTrue(np.allclose(s, [2,3,4,0,0,0,0,0]))
        s = overlapAdd(sequence, element, t=4, copy=True)
        self.assertTrue(np.allclose(s, [0,0,0,1,2,3,4,0]))
        s = overlapAdd(sequence, element, t=7, copy=True)
        self.assertTrue(np.allclose(s, [0,0,0,0,0,0,1,2]))
        
        sequence = np.zeros(8)
        element = np.arange(1,5)
        s = overlapAdd(sequence, element, t=-6, copy=True)
        self.assertTrue(np.allclose(s, sequence))
        s = overlapAdd(sequence, element, t=-20, copy=True)
        self.assertTrue(np.allclose(s, sequence))
        
        sequence = np.zeros(8)
        element = np.arange(1,6)
        s = overlapAdd(sequence, element, t=-6, copy=True)
        self.assertTrue(np.allclose(s, sequence))
        s = overlapAdd(sequence, element, t=-20, copy=True)
        self.assertTrue(np.allclose(s, sequence))
        
    def test_overlapReplace(self):
        
        sequence = np.arange(1,9)
        element = np.zeros(4)
        s = overlapReplace(sequence, element, t=0, copy=True)
        self.assertTrue(np.allclose(s, [0,0,0,4,5,6,7,8]))
        s = overlapReplace(sequence, element, t=4, copy=True)
        self.assertTrue(np.allclose(s, [1,2,3,0,0,0,0,8]))
        s = overlapReplace(sequence, element, t=7, copy=True)
        self.assertTrue(np.allclose(s, [1,2,3,4,5,6,0,0]))
        
        sequence = np.arange(1,9)
        element = np.zeros(5)
        s = overlapReplace(sequence, element, t=0, copy=True)
        self.assertTrue(np.allclose(s, [0,0,0,4,5,6,7,8]))
        s = overlapReplace(sequence, element, t=4, copy=True)
        self.assertTrue(np.allclose(s, [1,2,0,0,0,0,0,8]))
        s = overlapReplace(sequence, element, t=7, copy=True)
        self.assertTrue(np.allclose(s, [1,2,3,4,5,0,0,0]))
        
        sequence = np.arange(1,9)
        element = np.zeros(4)
        s = overlapReplace(sequence, element, t=-6, copy=True)
        self.assertTrue(np.allclose(s, sequence))
        s = overlapReplace(sequence, element, t=-20, copy=True)
        self.assertTrue(np.allclose(s, sequence))
        
        sequence = np.arange(1,9)
        element = np.zeros(5)
        s = overlapReplace(sequence, element, t=-6, copy=True)
        self.assertTrue(np.allclose(s, sequence))
        s = overlapReplace(sequence, element, t=-20, copy=True)
        self.assertTrue(np.allclose(s, sequence))
                
if __name__ == '__main__':
    unittest.main()
