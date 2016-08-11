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
This example implements and tests expm1 operator  - ie. exp(x) - 1, which is equivalent to that defined for numpy.
"""

import unittest
import numpy as np
import opveclib as ops
import tensorflow as tf

@ops.operator()
def expm1(x):
    """
    Define the operator function.

    :param x: The input tensor
    :return: Element-wise exp(x) - 1
    """

    output = ops.output_like(x)
    pos = ops.position_in(x.shape)

    e = ops.exp(x[pos])
    with ops.if_(e == 1.0):
        output[pos] = x[pos]
    with ops.elif_ ((e - 1.0) == -1.0):
        output[pos] = -1.0
    with ops.else_():
        output[pos] = (e - 1.0) * x[pos]/ops.log(e)

    return output


class TestExpm1(unittest.TestCase):
    ops.clear_op_cache()
    def test(self):
        """
        Test the correctness of ovl operator vs numpy implementation
        """

        b = np.array([1e-10, -1e-10, 0.0], dtype=np.float64)
        np_res = np.expm1(b)
        ovl_res = ops.evaluate(expm1(b))
        ops.logger.debug(u'numpy: ' + str(np_res) + u' ovl: ' + str(ovl_res))
        assert np.allclose(np_res, ovl_res, rtol=0, atol=1e-20)
        if ops.cuda_enabled:
            assert np.allclose(np.expm1(b), ops.evaluate(expm1(b), target_language='cuda'), rtol=0, atol=1e-20)


        # test performance
        import timeit
        import time
        logger = ops.logger
        iters = 10
        X = np.random.uniform(0, 1, size=(10000, 1000))
        ref = np.expm1(X)
        # timeit returns seconds for 'number' iterations. For 10 iterations, multiply by 100 to get time in ms
        np_time = 100 * timeit.timeit('np.expm1(X)',
                                      setup='import numpy as np; X = np.random.uniform(0, 1, size=(10000, 1000))',
                                      number=iters)
        logger.debug(u'Best numpy time (ms): ' + str(np_time))
        expm1Op = expm1(X)
        ovl_cpp, prof_cpp = ops.profile(expm1Op, target_language='cpp', profiling_iterations=iters, opt_level=0)
        assert np.allclose(ref, ovl_cpp, rtol=0, atol=1e-15)
        ovl_cpp_time = np.min(list(prof_cpp.values())[0])
        logger.debug(u'Best ovl cpp time (ms): ' + str(ovl_cpp_time))
        if ops.cuda_enabled:
            ovl_cuda, prof_cuda = ops.profile(expm1Op, target_language='cuda', profiling_iterations=iters, opt_level=0)
            assert np.allclose(ref, ovl_cuda, rtol=0, atol=1e-15)
            ovl_cuda_time = np.min(list(prof_cuda.values())[0])
            logger.debug(u'Best ovl cuda time  (ms): ' + str(ovl_cuda_time))

        # OVL-TF integration
        # ensure TF runs on GPU
        test_config=tf.ConfigProto(allow_soft_placement=False)
        test_config.graph_options.optimizer_options.opt_level = -1
        ones = np.ones_like(X)
        if ops.cuda_enabled:
            devices = ['/cpu:0', '/gpu:0']
        else:
            devices = ['/cpu:0']
        with tf.Session(config=test_config) as sess:
           for dev_string in devices:
                with tf.device(dev_string):
                    expm1_tf = ops.as_tensorflow(expm1Op)
                    sess.run(tf.initialize_all_variables())
                    expm1_tf_result = sess.run(expm1_tf)
                    prof_ovl = np.zeros(iters)
                    for i in range(iters):
                        t0 = time.time()
                        sess.run(expm1_tf.op)
                        t1 = time.time()
                        prof_ovl[i] = t1 - t0
                    tf_ovl_time = np.min(prof_ovl) * 1000.00
                    logger.debug(u'Best tf + ovl time  (ms) on ' + dev_string + ' :' + str(tf_ovl_time))
                    assert np.allclose(ref, expm1_tf_result, rtol=0, atol=1e-15)

                    # TF exp - 1
                    tf_out = tf.exp(X) - ones
                    tf_result = tf_out.eval()
                    assert np.allclose(ref, tf_result, rtol=0, atol=1e-15)
                    prof_tf = np.zeros(iters)
                    for i in range(iters):
                        t0 = time.time()
                        sess.run(tf_out.op)
                        t1 = time.time()
                        prof_tf[i] = t1 - t0
                    tf_time = np.min(prof_tf) * 1000.00
                    logger.debug(u'Best tf expm1 time  (ms) on ' + dev_string + ' :' + str(tf_time))
        sess.close()
