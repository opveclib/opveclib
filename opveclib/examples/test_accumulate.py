# Copyright 2016 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
# the specific language governing permissions and limitations under the License.


import unittest
import numpy as np
import opveclib as ops
from sys import _getframe

class Accumulate(ops.Operator):
    def op(self, x, inner_fcn, axis):
        """
        Define the operator function
        :param x: The input tensor
        :param inner_fcn: a lambda function to be applied for accumulation
        :param axis: The axis across which accumulation will be applied
        :return:
        """

        # ensure that the axis parameter makes sense
        assert isinstance(axis, int)
        assert axis >= 0
        assert axis < x.rank

        # Define the workgroup shape. Here we use a single worker to perform the accumulation across the
        # accumulation axis. The workgroup shape is then the size of all other axes with accumulation axis removed.
        workgroup_shape = []
        for cur_dim, num_elements in enumerate(x.shape):
            if cur_dim == axis:
                pass
            else:
                workgroup_shape.append(num_elements)
        pos = ops.position_in(workgroup_shape)

        # Define the accumulated output to be the same type as the input
        out = ops.output_like(x)

        # Define a function for determining the index of the input tensor as a function of accumulation axis position
        # and the current worker position. This is equal to the worker position with the accumulation axis position
        # inserted where it should be in the indexing order.
        def resolve_position(axis_n):
            cur_pos = []
            offset = 0
            for cur_dim in range(x.rank):
                if cur_dim == axis:
                    cur_pos.append(axis_n)
                    offset = 1
                else:
                    cur_pos.append(pos[cur_dim-offset])
            return cur_pos

        # initialize accumulator to be the first position along the accumulation axis
        initial_value = x[resolve_position(0)]
        accum = ops.variable(initial_value, x.dtype)
        out[resolve_position(0)] = accum

        # use this worker to iterate over and accumulate the rest of the elements in the accumulation axis
        for i in ops.arange(1, x.shape[axis]):
            accum <<= inner_fcn(accum, x[resolve_position(i)])
            out[resolve_position(i)] = accum

        return out


def cumsum(x, axis=0):
    """
    Define the cumsum operator by defining the inner_fcn to be addition
    :param x: the input tensor
    :param axis: the accumulation axis
    :return: the cumulative sum across the accumulation axis
    """
    # Define the op. Note that constants (inner_fcn and axis) must be passed as keyword arguments.
    op = Accumulate(x, inner_fcn=lambda arg1, arg2: arg1+arg2, axis=axis)
    return op


def cumprod(x, axis=0):
    """
    Define the cumprod operator by defining the inner_fcn to be multiplication
    :param x: the input tensor
    :param axis: the accumulation axis
    :return: the cumulative product across the accumulation axis
    """
    # Define the op. Note that constants (inner_fcn and axis) must be passed as keyword arguments.
    op = Accumulate(x, inner_fcn=lambda arg1, arg2: arg1*arg2, axis=axis)
    return op


class TestAccumulate(unittest.TestCase):
    def test(self):
        """
        Test the outputs of the operators to make sure they are consistent with the numpy implementation
        """
        print('*** Running Test: ' + self.__class__.__name__ + ' function: ' + _getframe().f_code.co_name)
        a = np.random.random((5, 5, 5))
        assert np.allclose(np.cumsum(a, axis=0), cumsum(a, axis=0).evaluate_c())
        assert np.allclose(np.cumsum(a, axis=1), cumsum(a, axis=1).evaluate_c())
        assert np.allclose(np.cumsum(a, axis=2), cumsum(a, axis=2).evaluate_c())

        assert np.allclose(np.cumprod(a, axis=0), cumprod(a, axis=0).evaluate_c())
        assert np.allclose(np.cumprod(a, axis=1), cumprod(a, axis=1).evaluate_c())
        assert np.allclose(np.cumprod(a, axis=2), cumprod(a, axis=2).evaluate_c())

        if ops.cuda_enabled:
            assert np.allclose(np.cumsum(a, axis=0), cumsum(a, axis=0).evaluate_cuda())
            assert np.allclose(np.cumsum(a, axis=1), cumsum(a, axis=1).evaluate_cuda())
            assert np.allclose(np.cumsum(a, axis=2), cumsum(a, axis=2).evaluate_cuda())

            assert np.allclose(np.cumprod(a, axis=0), cumprod(a, axis=0).evaluate_cuda())
            assert np.allclose(np.cumprod(a, axis=1), cumprod(a, axis=1).evaluate_cuda())
            assert np.allclose(np.cumprod(a, axis=2), cumprod(a, axis=2).evaluate_cuda())


if __name__ == '__main__':
    unittest.main()
