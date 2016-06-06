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
import sys
import getopt
import time
import string
import gc
import numpy as np
import tensorflow as tf
import opveclib as ops
from tensorflow.python.ops import math_ops


# performance test for comparing throughput of a multi-tensor add operation
class sum_of_squares(ops.Operator):
    def op(self, input0, input1, input2):
        sum = ops.output_like(input0)
        pos = ops.position_in(input0.shape)

        a = input0[pos]
        b = input1[pos]
        c = input2[pos]
        sum[pos] = ops.sqrt(a*a + b*b + c*c)

        return sum


def main(argv):
    # default options
    device = 'CPU'
    dev_string = '/cpu:0'
    length = int(1e5)
    method = 'OVL'
    iters = 100
    csv = False

    try:
        opts, args = getopt.getopt(argv, "d:l:m:i:csv")
    except getopt.GetoptError:
        raise ValueError('perf_sum_of_squares.py -d <[CPU, GPU]> -l <input length> -m <[TF, OVL]> -i <iterations> -csv')

    for opt, arg in opts:
        if opt in '-d':
            device = arg
            if device in 'CPU':
                dev_string = '/cpu:0'
            elif device in 'GPU':
                if ops.cuda_enabled:
                    dev_string = '/gpu:0'
                else:
                    raise ValueError('cannot run on GPU - cuda not installed')
            else:
                raise ValueError('device must be CPU or GPU')
        elif opt in '-l':
            length = int(float(arg))
        elif opt in '-m':
            if arg not in ['TF', 'OVL']:
                raise ValueError('method must be TF or OVL')
            method = arg
        elif opt in '-i':
            iters = int(float(arg))
        elif opt in '-csv':
            csv = True

    rng = np.random.RandomState()
    in0 = rng.uniform(-1, 1, length).astype(np.float32)
    in1 = rng.uniform(-1, 1, length).astype(np.float32)
    in2 = rng.uniform(-1, 1, length).astype(np.float32)
    ref_out = np.sqrt(in0*in0 + in1*in1 + in2*in2)

    with tf.Session() as sess:
        with tf.device(dev_string):
            in0_tf = tf.constant(in0, dtype=tf.float32)
            in1_tf = tf.constant(in1, dtype=tf.float32)
            in2_tf = tf.constant(in2, dtype=tf.float32)

            if method == 'TF':
                out = tf.sqrt(math_ops.add_n([in0_tf*in0_tf, in1_tf*in1_tf, in2_tf*in2_tf]))
            elif method == 'OVL':
                out_op = sum_of_squares(in0_tf, in1_tf, in2_tf)
                out = out_op.as_tensorflow(cuda_threads_per_block=256)
            else:
                raise ValueError

            init = tf.initialize_all_variables()
            sess.run(init)
            out_eval = out.eval()
            assert np.allclose(out_eval, ref_out, atol=1e-06)

            times = []
            gc.disable()
            init_time = time.clock()
            total_time = 0
            i = 0
            while total_time < 100 and i < iters:
                t0 = time.clock()
                sess.run(out.op)
                t1 = time.clock()
                times.append(t1-t0)
                total_time = t1-total_time
                i += 1

            gc.enable()
            times = np.array(times)*1000
            minimum = np.min(times)
            p20 = np.percentile(times, 20)
            med = np.median(times)
            p80 = np.percentile(times, 80)
            maximum = np.max(times)

    if csv:
        out_string = '${minimum}, ${p20}, ${med}, ${p80}, ${maximum}'
        print(string.Template(out_string).substitute(locals()))
    else:
        out_string = 'device: ${device}\n' \
                     'method: ${method}\n' \
                     'length: ${length}\n' \
                     'minimum: ${minimum}\n' \
                     '20th percentile: ${p20}\n' \
                     'median: ${med}\n' \
                     '80th percentile: ${p80}\n' \
                     'maximum: ${maximum}'
        print(string.Template(out_string).substitute(locals()))


if __name__ == "__main__":
    main(sys.argv[1:])
