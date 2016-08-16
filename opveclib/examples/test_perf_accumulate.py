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
from .test_accumulate import cumsum


class TestAccumulatePerf(unittest.TestCase):
    ovl.clear_op_cache()
    def test_performance(self):
        """
        test the performance vs. numpy running standalone and from tensorflow
        based on tensorflow issue 813
        https://github.com/tensorflow/tensorflow/issues/813
        """
        import tensorflow as tf
        import timeit
        import time
        logger = ovl.logger
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
        ovl_cpp, prof_cpp = ovl.profile(cumsumOp, target_language='cpp', profiling_iterations=iters, opt_level=0)
        assert np.allclose(ref, ovl_cpp)
        ovl_cpp_time = np.min(list(prof_cpp.values())[0])
        logger.debug(u'Best ovl cpp time (ms): ' + str(ovl_cpp_time))
        if ovl.cuda_enabled:
            ovl_cuda, prof_cuda = ovl.profile(cumsumOp, target_language='cuda', profiling_iterations=iters, opt_level=0)
            assert np.allclose(ref, ovl_cuda)
            ovl_cuda_time = np.min(list(prof_cuda.values())[0])
            logger.debug(u'Best ovl cuda time  (ms): ' + str(ovl_cuda_time))

        # OVL-TF integration
        # ensure TF runs on GPU
        test_config=tf.ConfigProto(allow_soft_placement=False)
        test_config.graph_options.optimizer_options.opt_level = -1
        if ovl.cuda_enabled:
            devices = ['/cpu:0', '/gpu:0']
        else:
            devices = ['/cpu:0']
        with tf.Session(config=test_config) as sess:
           for dev_string in devices:
                with tf.device(dev_string):
                    cumsum_tf = ovl.as_tensorflow(cumsumOp)
                    sess.run(tf.initialize_all_variables())
                    cumsum_tf_result = sess.run(cumsum_tf)
                    prof_ovl = np.zeros(iters)
                    for i in range(iters):
                        t0 = time.time()
                        sess.run(cumsum_tf.op)
                        t1 = time.time()
                        prof_ovl[i] = t1 - t0
                    tf_ovl_time = np.min(prof_ovl) * 1000.00
                    logger.debug(u'Best tf + ovl time  (ms) on ' + dev_string + ' :' + str(tf_ovl_time))
                    assert np.allclose(ref, cumsum_tf_result)

                    # TF cumsum
                    tf_out = tf.cumsum(X, axis=0, exclusive=False, reverse=False)
                    tf_result = tf_out.eval()
                    assert np.allclose(ref, tf_result)
                    prof_tf = np.zeros(iters)
                    for i in range(iters):
                        t0 = time.time()
                        sess.run(tf_out.op)
                        t1 = time.time()
                        prof_tf[i] = t1 - t0
                    tf_time = np.min(prof_tf) * 1000.00
                    logger.debug(u'Best tf cumsum time  (ms) on ' + dev_string + ' :' + str(tf_time))
        sess.close()
