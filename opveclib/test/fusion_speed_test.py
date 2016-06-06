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
import datetime
import itertools
import numpy as np
import tensorflow as tf
import opveclib as ops
# from memory_profiler import profile

# performance test for comparing throughput of a made up merge-able sub-graph
class FuseOp(ops.Operator):
    # output is the sum of the squares of each input
    def op(self, input0, input1, input2):
        pos = ops.position_in(input0.shape)
        out0 = ops.output_like(input0)
        a = input0[pos]
        b = input1[pos]
        c = input2[pos]
        d = a*a + b*b + c*c
        out0[pos] = d
        return out0


# TensorFlow standalone test
# @profile
def run_tf(dev_string, iters, in0, in1, in2, out0_np):
    with tf.Session() as sess:
        with tf.device(dev_string):
            in0_tf = tf.constant(in0, dtype=tf.float32)
            in1_tf = tf.constant(in1, dtype=tf.float32)
            in2_tf = tf.constant(in2, dtype=tf.float32)
            sq0 = tf.square(in0_tf)
            sq1 = tf.square(in1_tf)
            sq2 = tf.square(in2_tf)
            sum1 = sq0 + sq1
            out0 = sum1 + sq2

        out0_tf = sess.run([out0])
        t0 = time.clock()
        for dummy in itertools.repeat(None, iters):
            sess.run([out0.op])
        t1 = time.clock()
        tf_time = (t1-t0)/float(iters) * 1000.00
    sess.close()
    assert np.allclose(out0_tf, out0_np)
    return tf_time

# @profile
def run_integrated(dev_string, iters, in0, in1, in2, out0_np):
    with tf.Session() as sess:
        with tf.device(dev_string):
            in0_tf = tf.constant(in0, dtype=tf.float32)
            in1_tf = tf.constant(in1, dtype=tf.float32)
            in2_tf = tf.constant(in2, dtype=tf.float32)

            op_tf = FuseOp(in0_tf, in1_tf, in2_tf)
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
    assert np.allclose(out0_int, out0_np)
    return to_integrated_time

def main(argv):
    # default options
    device = 'CPU'
    max_len = int(1e5)
    csv = False
    # parse command line
    try:
        opts, args = getopt.getopt(argv,"d:l:csv")
    except getopt.GetoptError:
        print('fusion_speed_test.py -d <[CPU, GPU]> -l <input length> -csv')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-d"):
            device = arg
        if opt in ("-l"):
            max_len = int(float(arg))
        if opt in ("-csv"):
            csv = True

    rng = np.random.RandomState()
    iters = 100

    if device == 'CPU':
        dev_string = '/cpu:0'
    elif device == 'GPU':
        if ops.cuda_enabled:
            dev_string = '/gpu:0'
        else:
            raise ValueError('cannot run on GPU - cuda not installed')
    else:
        raise ValueError('device must be CPU or GPU')

    # if we are generating a csv file, loop over each input size from 1e1 to 1e8
    if csv:
        len = 10
        max_len = 1e8
        today = datetime.date.today()
        print(today)
    else:
        len = max_len
    while len <= max_len:
        in0 = rng.uniform(-1, 1, len).astype(np.float32)
        in1 = rng.uniform(-1, 1, len).astype(np.float32)
        in2 = rng.uniform(-1, 1, len).astype(np.float32)

        # Numpy test
        t0 = time.clock()
        for dummy in itertools.repeat(None, iters):
            out0_np = in0*in0 + in1*in1 + in2*in2
            t1 = time.clock()
        np_time = (t1-t0)/float(iters) * 1000.00

        # run tensorflow standalone test
        # TF fails with  ValueError: GraphDef cannot be larger than 2GB at input len 1e8
        tf_time = 0
        if len <= int(1e7):
            tf_time = run_tf(dev_string, iters, in0, in1, in2, out0_np)

        # run OVL/TensorFlow integration test
        to_integrated_time = 0
        to_integrated_time = run_integrated(dev_string, iters, in0, in1, in2, out0_np)

        # OVL standalone test
        op = FuseOp(in0, in1, in2)
        if device == 'CPU':
            def direct_eval(): return op.evaluate_c()
        elif device == 'GPU':
            def direct_eval(): return op.evaluate_cuda()
        else:
            raise ValueError('device must be CPU or GPU')

        # OVL direct is very slow and measurement is meaningless on GPU
        if device == 'CPU':
            direct_eval()
            t0 = time.clock()
            for dummy in itertools.repeat(None, iters):
                out0_to = direct_eval()
            t1 = time.clock()
            to_direct_time = (t1-t0)/float(iters) * 1000.00
            assert np.allclose(out0_to, out0_np)

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
            if device == 'CPU':
                print(('OVL direct library compute time (ms): ', to_direct_time))

        # end loop
        len *= 10

if __name__ == "__main__":
    main(sys.argv[1:])
