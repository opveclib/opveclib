from __future__ import print_function
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
import logging
import opveclib as ops


@ops.operator()
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

    # initialize accumulator to be the first element along the accumulation axis
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
    # def test(self):
    #     """
    #     Test the outputs of the operators to make sure they are consistent with the numpy implementation
    #     """
    #
    #     a = np.random.random((5, 5, 5))
    #     logging.log(logging.DEBUG, u'Testing C')
    #     assert np.allclose(np.cumsum(a, axis=0), ops.evaluate(cumsum(a, axis=0), target_language='cpp'))
    #     assert np.allclose(np.cumsum(a, axis=1), ops.evaluate(cumsum(a, axis=1), target_language='cpp'))
    #     assert np.allclose(np.cumsum(a, axis=2), ops.evaluate(cumsum(a, axis=2), target_language='cpp'))
    #
    #     assert np.allclose(np.cumprod(a, axis=0), ops.evaluate(cumprod(a, axis=0), target_language='cpp'))
    #     assert np.allclose(np.cumprod(a, axis=1), ops.evaluate(cumprod(a, axis=1), target_language='cpp'))
    #     assert np.allclose(np.cumprod(a, axis=2), ops.evaluate(cumprod(a, axis=2), target_language='cpp'))
    #
    #     if ops.cuda_enabled:
    #         logging.log(logging.DEBUG, u'Testing CUDA')
    #         assert np.allclose(np.cumsum(a, axis=0), ops.evaluate(cumsum(a, axis=0), target_language='cuda'))
    #         assert np.allclose(np.cumsum(a, axis=1), ops.evaluate(cumsum(a, axis=1), target_language='cuda'))
    #         assert np.allclose(np.cumsum(a, axis=2), ops.evaluate(cumsum(a, axis=2), target_language='cuda'))
    #
    #         assert np.allclose(np.cumprod(a, axis=0), ops.evaluate(cumprod(a, axis=0), target_language='cuda'))
    #         assert np.allclose(np.cumprod(a, axis=1), ops.evaluate(cumprod(a, axis=1), target_language='cuda'))
    #         assert np.allclose(np.cumprod(a, axis=2), ops.evaluate(cumprod(a, axis=2), target_language='cuda'))

    def test_performance(self):
        """
        test the performance vs. numpy running standalone and from tensorflow
        based on tensorflow issue 813
        https://github.com/tensorflow/tensorflow/issues/813
        """
        import tensorflow as tf
        import timeit
        import time
        logger = logging.getLogger('cumsum')
        logger.setLevel(logging.DEBUG)
        iters = 10
        X = np.random.uniform(0, 1, size=(10000, 1000))
        # note, np.cumsum fails with memory error at input size 10 ^^ 6
        ref = np.cumsum(X, axis=0)
        # timeit returns seconds for 'number' iterations. For 10 iterations, multiply by 100 to get time in ms
        np_time = 100 * timeit.timeit('np.cumsum(X, axis=0)',
                                      setup='import numpy as np; X = np.random.uniform(0, 1, size=(10000, 1000))',
                                      number=iters)
        logger.debug(u'Best numpy time (ms): ' + str(np_time))
        cumsumOp = cumsum(X, axis=0)
        ovl_cpp, prof_cpp = ops.profile(cumsumOp, target_language='cpp', profiling_iterations=iters)
        assert np.allclose(ref, ovl_cpp)
        ovl_cpp_time = np.min(list(prof_cpp.values())[0])
        logger.debug(u'Best ovl cpp time (ms): ' + str(ovl_cpp_time))
        if ops.cuda_enabled:
            ovl_cuda, prof_cuda = ops.profile(cumsumOp, target_language='cuda', profiling_iterations=iters)
            assert np.allclose(ref, ovl_cuda)
            ovl_cuda_time = np.min(list(prof_cuda.values())[0])
            logger.debug(u'Best ovl cuda time  (ms): ' + str(ovl_cuda_time))

        # OVL-TF integration
        # ensure TF runs on GPU
        test_config=tf.ConfigProto(allow_soft_placement=False)
        test_config.graph_options.optimizer_options.opt_level = -1
        devices = ['/cpu:0', '/gpu:0']
        with tf.Session(config=test_config) as sess:
           for dev_string in devices:
                with tf.device(dev_string):
                    cumsum_tf = ops.as_tensorflow(cumsumOp)
                    sess.run(tf.initialize_all_variables())
                    cumsum_tf_result = sess.run(cumsum_tf)
                    prof_tf = np.zeros(iters)
                    for i in range(iters):
                        t0 = time.time()
                        sess.run(cumsum_tf.op)
                        t1 = time.time()
                        prof_tf[i] = t1 - t0
                    tf_time = np.min(prof_tf) * 1000.00
                    logger.debug(u'Best tf + ovl time  (ms) on ' + dev_string + ' :' + str(tf_time))
                    assert np.allclose(ref, cumsum_tf_result)
        sess.close()


if __name__ == '__main__':
    # ops.clear_op_cache()
    unittest.main()
