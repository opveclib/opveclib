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
import sys, getopt
import time
import itertools
import numpy as np
import tensorflow as tf
import opveclib as ops
from tensorflow.python.ops import math_ops


# performance test for comparing throughput of a multi-tensor add operation
class AggregateOp(ops.Operator):
    def op(self, input0, input1, input2, input3, input4):
        sum = ops.output_like(input0)
        pos = ops.position_in(input0.shape)
        sum[pos] = input0[pos] + input1[pos] + input2[pos] + input3[pos] + input4[pos]
        return sum


def main(argv):
    # default options
    device = 'CPU'
    len = int(1e5)
    csv = False
    tf_time = 0
    # parse command line
    try:
        opts, args = getopt.getopt(argv,"d:l:csv")
    except getopt.GetoptError:
        print('aggregate_speed_test.py -d <[CPU, GPU]> -l <input length> -csv')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-d"):
            device = arg
        if opt in ("-l"):
            len = int(float(arg))
        if opt in ("-csv"):
            csv = True

    rng = np.random.RandomState()
    iters = 100
    in0 = rng.uniform(-1, 1, len).astype(np.float32)
    in1 = rng.uniform(-1, 1, len).astype(np.float32)
    in2 = rng.uniform(-1, 1, len).astype(np.float32)
    in3 = rng.uniform(-1, 1, len).astype(np.float32)
    in4 = rng.uniform(-1, 1, len).astype(np.float32)

    if device == 'CPU':
        dev_string = '/cpu:0'
    elif device == 'GPU':
        if ops.cuda_enabled:
            dev_string = '/gpu:0'
        else:
            raise ValueError('cannot run on GPU - cuda not installed')
    else:
        raise ValueError('device must be CPU or GPU')

    # Numpy test
    t0 = time.clock()
    for dummy in itertools.repeat(None, iters):
        out0_np = in0 + in1 + in2 + in3 + in4
    t1 = time.clock()
    np_time = (t1-t0)/float(iters) * 1000.00

    # TensorFlow standalone test
    # TF fails with  ValueError: GraphDef cannot be larger than 2GB at input len 1e8
    if len <= int(1e7):
        with tf.Session() as sess:
            with tf.device(dev_string):
                input_list = []
                input_list.append(tf.constant(in0, dtype=tf.float32))
                input_list.append(tf.constant(in1, dtype=tf.float32))
                input_list.append(tf.constant(in2, dtype=tf.float32))
                input_list.append(tf.constant(in3, dtype=tf.float32))
                input_list.append(tf.constant(in4, dtype=tf.float32))
                out0 = math_ops.add_n(input_list)

            out0_tf = sess.run([out0])
            t0 = time.clock()
            for dummy in itertools.repeat(None, iters):
                sess.run([out0.op])
            t1 = time.clock()
            tf_time = (t1-t0)/float(iters) * 1000.00
        sess.close()
        assert np.allclose(out0_tf, out0_np)

    # OVL/TensorFlow integration test
    with tf.Session() as sess:
        with tf.device(dev_string):
            in0_tf = tf.constant(in0, dtype=tf.float32)
            in1_tf = tf.constant(in1, dtype=tf.float32)
            in2_tf = tf.constant(in2, dtype=tf.float32)
            in3_tf = tf.constant(in3, dtype=tf.float32)
            in4_tf = tf.constant(in4, dtype=tf.float32)

            op_tf = AggregateOp(in0_tf, in1_tf, in2_tf, in3_tf, in4_tf, clear_cache=True)
            out0_ops = op_tf.as_tensorflow()
            init = tf.initialize_all_variables()
        sess.run(init)
        out0_int = sess.run(out0_ops)
        t0 = time.clock()
        for dummy in itertools.repeat(None, iters):
            sess.run(out0_ops.op)
        t1 = time.clock()
        to_integrated_time = (t1-t0)/float(iters) * 1000.00
    sess.close()
    assert np.allclose(out0_int, out0_np, atol=1e-06)
    # default absolute tolerance of 1e-08 fails on GPU at input len 1e5
    # it fails on 1e-7 on input len = 1e8

    if csv:
        print(str(len) + ',' \
              + str(tf_time) + ',' \
              + str(to_integrated_time) + ',' \
              + str(np_time) + ',' + device)
    else:
        print('Device: ' + device + ' input length: ' + str(len))
        print(('Standalone TensorFlow compute time (ms): ', tf_time))
        print(('OVL/TensorFlow integrated compute time (ms): ', to_integrated_time))
        print(('Numpy compute time (ms): ', np_time))

if __name__ == "__main__":
    main(sys.argv[1:])
