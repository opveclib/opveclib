# Copyright 2016 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import time
import numpy as np
import tensorflow as tf

num_batches = 20
memory_length = 200
forget_bias = 0
use_ops = False
time_grad = False

with tf.Session() as sess:
    concat = tf.random_normal((num_batches, 4*memory_length))
    c = tf.random_normal((num_batches, memory_length))

    if use_ops:
        from LSTMP import LSTMP
        op = LSTMP(concat, c, forget_bias=forget_bias)
        new_c, new_h = op.as_tensorflow()
    else:
        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(1, 4, concat)

        new_c = c * tf.sigmoid(f + forget_bias) + tf.sigmoid(i) * tf.tanh(j)
        new_h = tf.tanh(new_c) * tf.sigmoid(o)

    if time_grad:
        grad = tf.gradients(new_h, c)[0]
        time_op = grad.op
    else:
        time_op = new_h.op

    init = tf.initialize_all_variables()
    sess.run(init)

    delta_t = []
    for i in range(100000):
        t0 = time.time()
        sess.run([time_op])
        t1 = time.time()
        delta_t.append(t1-t0)
    np.array(delta_t)

    print(np.min(delta_t))
