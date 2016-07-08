# Copyright 2016 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
# the specific language governing permissions and limitations under the License.

from __future__ import print_function
import unittest
import numpy as np
from sys import _getframe
from ..operator import Operator, evaluate
from ..expression import output_like, position_in, absolute, logical_not, arctan, arccos, arcsin, \
    cos, cosh, sin, sinh, tan, tanh, exp, log, log10, sqrt, ceil, floor
from ..local import cuda_enabled, clear_op_cache


def gen(input_tensor, ops_func, np_func, cuda_tolerance=None):
    class UnaryMath(Operator):
        def op(self, input0):

            output = output_like(input0)
            pos = position_in(input0.shape)

            output[pos] = ops_func(input0[pos])

            return output

    op = UnaryMath(input_tensor)
    op_c = evaluate(op, target_language='cpp')
    assert np.allclose(op_c, np_func(input_tensor))

    if cuda_enabled:
        op_cuda = evaluate(op, target_language='cuda')
        if cuda_tolerance is None:
            assert np.allclose(op_cuda, np_func(input_tensor))
        else:
            assert np.allclose(op_cuda, np_func(input_tensor),
                               atol=cuda_tolerance['atol'], rtol=cuda_tolerance['rtol'])


class TestUnaryMath(unittest.TestCase):
    def test_unary_float(self):
        print('*** Running Test: ' + self.__class__.__name__ + ' function: ' + _getframe().f_code.co_name)
        self.unary(np.float32)
    test_unary_float.regression = 1

    def test_unary_double(self):
        print('*** Running Test: ' + self.__class__.__name__ + ' function: ' + _getframe().f_code.co_name)
        self.unary(np.float64)
    test_unary_double.regression = 1

    def test_abs(self):
        print('*** Running Test: ' + self.__class__.__name__ + ' function: ' + _getframe().f_code.co_name)
        length = 100
        rng = np.random.RandomState(1)

        rand = rng.uniform(-10, 10, length)

        gen(rand.astype(np.float32), lambda x: absolute(x), lambda x: np.absolute(x))
        gen(rand.astype(np.float64), lambda x: absolute(x), lambda x: np.absolute(x))
        rand = rng.randint(-2**7, 2**7-1, length).astype(np.int8)
        gen(rand, lambda x: absolute(x), lambda x: np.absolute(x))
        rand = rng.randint(-2**15, 2**15-1, length).astype(np.int16)
        gen(rand, lambda x: absolute(x), lambda x: np.absolute(x))
        rand = rng.randint(-2**31, 2**31-1, length).astype(np.int32)
        gen(rand, lambda x: absolute(x), lambda x: np.absolute(x))
        rand = rng.randint(-2**63, 2**63-1, length).astype(np.int64)
        gen(rand, lambda x: absolute(x), lambda x: np.absolute(x))
    test_abs.regression = 1

    def test_negate(self):
        print('*** Running Test: ' + self.__class__.__name__ + ' function: ' + _getframe().f_code.co_name)
        length = 100
        rng = np.random.RandomState(1)

        rand = rng.uniform(-10, 10, length)

        def neg(x):
            return -x

        types = [np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]
        for t in types:
            gen(rand.astype(t), neg, neg)
    test_negate.regression = 1

    def test_not(self):
        print('*** Running Test: ' + self.__class__.__name__ + ' function: ' + _getframe().f_code.co_name)
        length = 100
        rng = np.random.RandomState(1)
        rand = rng.randint(0, 2, length)

        def not_ops(x):
            return logical_not(x)

        def not_np(x):
            return np.logical_not(x)
        
        types = [np.float32, np.float64, np.int8, np.int16, np.int32, np.int64,
                 np.uint8, np.uint16, np.uint32, np.uint64]
        for t in types:
            gen(rand.astype(t), not_ops, not_np)
    test_not.regression = 1

    def unary(self, input_type):
        length = 100
        rng = np.random.RandomState(1)
        rand = rng.uniform(-10, 10, length).astype(input_type)
        rand_pos = np.abs(rand) + np.finfo(input_type).eps
        rand_one = rng.uniform(-1.0+np.finfo(input_type).eps, 1 - np.finfo(input_type).eps, length).astype(input_type)

        gen(rand_one, lambda x: arccos(x), lambda x: np.arccos(x))
        gen(rand_one, lambda x: arcsin(x), lambda x: np.arcsin(x))
        gen(3.14 / 2 * rand_one, lambda x: arctan(x), lambda x: np.arctan(x))
        gen(rand, lambda x: cos(x), lambda x: np.cos(x))
        gen(rand, lambda x: cosh(x), lambda x: np.cosh(x))
        gen(rand, lambda x: sinh(x), lambda x: np.sinh(x))
        gen(rand, lambda x: tanh(x), lambda x: np.tanh(x))
        gen(rand, lambda x: exp(x), lambda x: np.exp(x))
        gen(rand_pos, lambda x: log(x), lambda x: np.log(x))
        gen(rand_pos, lambda x: log10(x), lambda x: np.log10(x))
        gen(rand_pos, lambda x: sqrt(x), lambda x: np.sqrt(x))
        gen(rand, lambda x: ceil(x), lambda x: np.ceil(x))
        gen(rand, lambda x: floor(x), lambda x: np.floor(x))

        # When using single precision floats and the fast-math compiler flag, CUDA is
        # less precise for the sin and tan functions than the default tolerance of np.allclose
        # Set the tolerances higher in this special case.
        if rand.dtype == np.float32:
            gen(rand, lambda x: sin(x), lambda x: np.sin(x),
                cuda_tolerance={'atol': 1e-6, 'rtol': 1e-4})
        else:
            gen(rand, lambda x: sin(x), lambda x: np.sin(x))

        if rand.dtype == np.float32:
            gen(rand, lambda x: tan(x), lambda x: np.tan(x),
                cuda_tolerance={'atol': 1e-6, 'rtol': 1e-4})
        else:
            gen(rand, lambda x: tan(x), lambda x: np.tan(x))

if __name__ == '__main__':
    clear_op_cache()
    unittest.main()
