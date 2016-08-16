# Copyright 2016 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
# the specific language governing permissions and limitations under the License.

"""
This example implements and tests a generic accumulation operator which applies and accumulates a lambda function
across a specified axis of the input tensor. This generic accumulation operator is used to define the ``cumsum`` and
``cumprod`` functions, which are equivalent to those defined for numpy.
"""

import unittest
import numpy as np
import opveclib as ovl


@ovl.operator()
def accumulate(x, inner_fcn=None, axis=None):
    """
    Define the operator function.

    :param x: The input tensor
    :param inner_fcn: a lambda function to be applied for accumulation
    :param axis: The axis across which accumulation will be applied
    :return: The accumulated result
    """

    # assert that the axis parameter makes sense
    assert isinstance(axis, int)
    assert axis >= 0
    assert axis < x.rank

    # Define the workgroup shape. Here we use a single worker to perform the accumulation across the
    # accumulation axis. The workgroup shape is then the size of all other axes with accumulation axis removed.
    if x.rank is 1:
        workgroup_shape = [1]
    else:
        workgroup_shape = []
        for cur_dim, num_elements in enumerate(x.shape):
            if cur_dim == axis:
                pass
            else:
                workgroup_shape.append(num_elements)
    pos = ovl.position_in(workgroup_shape)

    # Define the accumulated output to be the same type as the input
    out = ovl.output_like(x)

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

    # initialize accumulator to be the first element along the accumulation axis
    initial_value = x[resolve_position(0)]
    accum = ovl.variable(initial_value, x.dtype)
    out[resolve_position(0)] = accum

    # use this worker to iterate over and accumulate the rest of the elements in the accumulation axis
    for i in ovl.arange(1, x.shape[axis]):
        accum <<= inner_fcn(accum, x[resolve_position(i)])
        out[resolve_position(i)] = accum

    return out


def cumsum(x, axis=0):
    """
    Define the cumsum operator by defining the inner_fcn to be addition

    .. math::
       y_{n_0,n_1,n_2,n_3} = \sum_{n_2=0}^{n_2} x_{n_0,n_1,n_2,n_3}

    where :math:`n_2` is the index over which to perform the addition.


    :param x: the input tensor
    :param axis: the accumulation axis
    :return: the cumulative sum across the accumulation axis

    :Examples:

    .. doctest::

        >>> import numpy
        >>> from opveclib import evaluate
        >>> from opveclib.examples import cumsum
        >>> a = numpy.arange(1, 6)
        >>> evaluate(cumsum(a))
        array([ 1,  3,  6, 10, 15])
        >>> b = numpy.arange(1,16).reshape(3,5)
        >>> evaluate(cumsum(b, axis=0))
        array([[ 1,  2,  3,  4,  5],
               [ 7,  9, 11, 13, 15],
               [18, 21, 24, 27, 30]])
    """
    # Define the op. Note that constants (inner_fcn and axis) must be passed as keyword arguments.
    op = accumulate(x, inner_fcn=lambda arg1, arg2: arg1+arg2, axis=axis)
    return op


def cumprod(x, axis=0):
    """
    Define the cumprod operator by defining the inner_fcn to be multiplication

    .. math::
        y_{n_0,n_1,n_2,n_3} = \prod_{n_2=0}^{n_2} x_{n_0,n_1,n_2,n_3}

    where :math:`n_2` is the index over which to perform the multipilication.

    :param x: the input tensor
    :param axis: the accumulation axis
    :return: the cumulative product across the accumulation axis

    :Examples:

    .. doctest::

        >>> import numpy
        >>> from opveclib import evaluate
        >>> from opveclib.examples import cumprod
        >>> a = numpy.arange(1, 6)
        >>> evaluate(cumprod(a))
        array([  1,   2,   6,  24, 120])
        >>> b = numpy.arange(1,16).reshape(3,5)
        >>> evaluate(cumprod(b, axis=0))
        array([[  1,   2,   3,   4,   5],
               [  6,  14,  24,  36,  50],
               [ 66, 168, 312, 504, 750]])
    """
    # Define the op. Note that constants (inner_fcn and axis) must be passed as keyword arguments.
    op = accumulate(x, inner_fcn=lambda arg1, arg2: arg1*arg2, axis=axis)
    return op


class TestAccumulate(unittest.TestCase):
    ovl.clear_op_cache()

    def test(self):
        """
        Test the outputs of the operators to make sure they are consistent with the numpy implementation
        """

        a = np.random.random((5, 5, 5))
        ovl.logger.debug(u'Testing C')
        assert np.allclose(np.cumsum(a, axis=0), ovl.evaluate(cumsum(a, axis=0), target_language='cpp'))
        assert np.allclose(np.cumsum(a, axis=1), ovl.evaluate(cumsum(a, axis=1), target_language='cpp'))
        assert np.allclose(np.cumsum(a, axis=2), ovl.evaluate(cumsum(a, axis=2), target_language='cpp'))

        assert np.allclose(np.cumprod(a, axis=0), ovl.evaluate(cumprod(a, axis=0), target_language='cpp'))
        assert np.allclose(np.cumprod(a, axis=1), ovl.evaluate(cumprod(a, axis=1), target_language='cpp'))
        assert np.allclose(np.cumprod(a, axis=2), ovl.evaluate(cumprod(a, axis=2), target_language='cpp'))

        if ovl.cuda_enabled:
            ovl.logger.debug(u'Testing CUDA')
            assert np.allclose(np.cumsum(a, axis=0), ovl.evaluate(cumsum(a, axis=0), target_language='cuda'))
            assert np.allclose(np.cumsum(a, axis=1), ovl.evaluate(cumsum(a, axis=1), target_language='cuda'))
            assert np.allclose(np.cumsum(a, axis=2), ovl.evaluate(cumsum(a, axis=2), target_language='cuda'))

            assert np.allclose(np.cumprod(a, axis=0), ovl.evaluate(cumprod(a, axis=0), target_language='cuda'))
            assert np.allclose(np.cumprod(a, axis=1), ovl.evaluate(cumprod(a, axis=1), target_language='cuda'))
            assert np.allclose(np.cumprod(a, axis=2), ovl.evaluate(cumprod(a, axis=2), target_language='cuda'))
