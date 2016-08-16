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
import tensorflow as tf
from opveclib.expression import position_in, output_like
from opveclib.operator import operator, as_tensorflow, evaluate
from opveclib.local import cuda_enabled, clear_op_cache


class TestIntegration(unittest.TestCase):
    clear_op_cache()

    def test_single_output(self):

        @operator()
        def add(x, y):
            pos = position_in(x.shape)
            out = output_like(x)
            out[pos] = x[pos] + y[pos]
            return out

        in0 = np.random.random(5).astype(np.float32)
        in1 = np.random.random(5).astype(np.float32)
        reference = 4*(in0 + in1)*(in0 + in1)

        test_config = tf.ConfigProto(allow_soft_placement=False)
        # Don't perform optimizations for tests so we don't inadvertently run
        # gpu ops on cpu
        test_config.graph_options.optimizer_options.opt_level = -1
        with tf.Session(config=test_config) as sess:
            with tf.device('/cpu:0'):
                a = in0*2
                b = in1*2
                c = as_tensorflow(add(a, b))
                squared = tf.square(c)
            if cuda_enabled:
                with tf.device('/gpu:0'):
                    a_gpu = in0*2
                    b_gpu = in1*2
                    c_gpu = as_tensorflow(add(a_gpu, b_gpu))
                    squared_gpu = tf.square(c_gpu)
                result, result_gpu = sess.run([squared, squared_gpu])
                assert np.allclose(reference, result_gpu)
            else:
                result = sess.run([squared])

        assert np.allclose(reference, result)

    def test_multiple_outputs(self):

        @operator()
        def multi_op(input0, input1, input2):
            pos = position_in(input0.shape)
            output0 = output_like(input0)
            output1 = output_like(input0)
            output2 = output_like(input0)

            a = input0[pos]
            b = input1[pos]
            c = input2[pos]
            d = a + b
            output0[pos] = d
            output1[pos] = d*c
            output2[pos] = d+c

            return output0, output1, output2

        rng = np.random.RandomState()
        in0 = rng.uniform(-1, 1, 5).astype(np.float32)
        in1 = rng.uniform(-1, 1, 5).astype(np.float32)
        in2 = rng.uniform(-1, 1, 5).astype(np.float32)

        np1 = in0*in0 + in1*in1
        np2 = np1*in2
        np3 = np1 + in2

        test_config = tf.ConfigProto(allow_soft_placement=False)
        # Don't perform optimizations for tests so we don't inadvertently run
        # gpu ops on cpu
        test_config.graph_options.optimizer_options.opt_level = -1
        with tf.Session(config=test_config) as sess:
            sq0 = tf.square(in0)
            sq1 = tf.square(in1)

            with tf.device('/cpu:0'):
                op = multi_op(sq0, sq1, in2)
                out0, out1, out2 = as_tensorflow(op)

            if cuda_enabled:
                with tf.device('/gpu:0'):
                    op_gpu = multi_op(sq0, sq1, in2)
                    out0_gpu, out1_gpu, out2_gpu = as_tensorflow(op_gpu)

                eval1, eval2, eval3, eval1_gpu, eval2_gpu, eval3_gpu = \
                    sess.run([out0, out1, out2, out0_gpu, out1_gpu, out2_gpu])
                assert np.allclose(eval1_gpu, np1)
                assert np.allclose(eval2_gpu, np2)
                assert np.allclose(eval3_gpu, np3)
            else:
                eval1, eval2, eval3 = sess.run([out0, out1, out2])

        assert np.allclose(eval1, np1)
        assert np.allclose(eval2, np2)
        assert np.allclose(eval3, np3)

    def test_4D(self):

        @operator()
        def sum_sq(input0, input1, input2):
            pos = position_in(input0.shape)
            out0 = output_like(input0)
            a = input0[pos]
            b = input1[pos]
            c = input2[pos]
            d = a*a + b*b + c*c
            out0[pos] = d
            return out0

        in0 = np.random.random((5, 4, 3, 2)).astype(np.float32)
        in1 = np.random.random((5, 4, 3, 2)).astype(np.float32)
        in2 = np.random.random((5, 4, 3, 2)).astype(np.float32)
        reference = np.square(in0) + np.square(in1) + np.square(in2)

        test_config = tf.ConfigProto(allow_soft_placement=False)
        # Don't perform optimizations for tests so we don't inadvertently run
        # gpu ops on cpu
        test_config.graph_options.optimizer_options.opt_level = -1
        op = sum_sq(in0, in1, in2)
        with tf.Session(config=test_config) as sess:
            with tf.device('/cpu:0'):
                out_cpu = as_tensorflow(op)
            if cuda_enabled:
                with tf.device('/gpu:0'):
                    out_gpu = as_tensorflow(op)
                result, result_gpu = sess.run([out_cpu, out_gpu])
                assert np.allclose(reference, result_gpu)
                # make sure the evaluate gives the correct results too
                assert np.allclose(evaluate(op, target_language='cuda'), reference)
            else:
                result = sess.run([out_cpu])

        assert np.allclose(reference, result)
        assert np.allclose(evaluate(op, target_language='cpp'), reference)
