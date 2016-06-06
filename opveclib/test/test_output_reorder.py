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
from ..operator import Operator
from ..expression import position_in, output_like


class TestOutputReturn(unittest.TestCase):
    # Operation should reorder the output io_index to be in the order with which they are returned, not the order
    # in which they are declared
    def test_return_reordering(self):
        print('*** Running Test: ' + self.__class__.__name__ + ' function: ' + _getframe().f_code.co_name)
        class ReorderOp(Operator):
            def op(self, input0):
                pos = position_in(input0.shape)

                # declare out of order
                output3 = output_like(input0)
                output0 = output_like(input0)
                output2 = output_like(input0)
                output1 = output_like(input0)

                input = input0[pos]

                output0[pos] = 2*input
                output1[pos] = 3*input
                output2[pos] = 4*input
                output3[pos] = 5*input

                # return in order
                return output0, output1, output2, output3

        a = np.random.random(5)
        op = ReorderOp(a, clear_cache=True)
        o0, o1, o2, o3 = op.evaluate_c()

        assert np.alltrue(np.equal(o0, 2*a))
        assert np.alltrue(np.equal(o1, 3*a))
        assert np.alltrue(np.equal(o2, 4*a))
        assert np.alltrue(np.equal(o3, 5*a))

    # Operation should throw and error when user tries to return the wrong number of outputs or the wrong type of value
    def test_output_error(self):
        print('*** Running Test: ' + self.__class__.__name__ + ' function: ' + _getframe().f_code.co_name)
        class TooFew(Operator):
            def op(self, input0):
                pos = position_in(input0.shape)

                # declare out of order
                output3 = output_like(input0)
                output0 = output_like(input0)
                output2 = output_like(input0)
                output1 = output_like(input0)

                input_value = input0[pos]

                output0[pos] = 2*input_value
                output1[pos] = 3*input_value
                output2[pos] = 4*input_value
                output3[pos] = 5*input_value

                # return in order
                return output0, output1

        a = np.random.random(5)
        try:
            TooFew(a, clear_cache=True)
        except ValueError:
            pass
        else:
            assert False

        class NoOutputs(Operator):
            def op(self, input0):
                pos = position_in(input0.shape)

                input_value = input0[pos]
                output0 = output_like(input0)
                output0[pos] = 2*input_value

        try:
            NoOutputs(a, clear_cache=True)
        except ValueError:
            pass
        else:
            assert False

        class BadReturn(Operator):
            def op(self, input0):
                pos = position_in(input0.shape)

                input_value = input0[pos]
                output0 = output_like(input0)
                output0[pos] = 2*input_value

                return input_value

        try:
            BadReturn(a, clear_cache=True)
        except TypeError:
            pass
        else:
            assert False


if __name__ == '__main__':
    unittest.main()
