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
import opveclib as ovl
import numpy as np
import tensorflow as tf

def kp_py(values, weights, capacity):
    """
    Solves the Knapsack problem using dynamic programming.
    :param values: The value of each item.
    :param weights: The weight or cost of taking an item.
    :param capacity: The capacity limitation of the knapsack in total weight or cost.
    :return: The maximum achievable value while maintaining the capacity limit.
    """
    assert len(values) == len(weights)
    n = len(values)
    accu = [0]*(capacity+1)
    for k in range(0, n):
        weight = int(weights[k])
        for c in range(capacity, weight-1, -1):
            value = accu[c-weight] + values[k]
            if accu[c] < value:
                accu[c] = value
    return accu[capacity]


@ovl.operator()
def kp_max(max_in, values, weights, index, capacity=None, n_work_per_worker=None):
    """
    Operator that computes

        max_out[i] = max(max_in[i-weights[index]]+values[index], max_in[i]) for i >= weights[index] and index >= 0.

    max_out is the output and max_in is the input maximum value for a given capacity having considered index items.

    :param max_in: Maximum values from the last iteration or all 0 initially.
    :param values: Value per item at index.
    :param weights: Weight per item at index.
    :param index: The index of the currently worked on item.
    :param capacity: The capacity limit.
    :param n_work_per_worker: Number of work (a max and sum) per worker. Typical values are > 50.
    :return: max_out: Maximum values after evaluating the above equation.
    """

    # TODO:(raudies@hpe.com): Once available in tensorflow use int for max_in, values, weights, and index.
    weight = ovl.cast(weights[index[0]], dtype=ovl.int64)
    value = values[index[0]]

    # Compute start and end indices for the items that i_worker processes.
    n_worker = int((capacity + 1 + n_work_per_worker-1) / n_work_per_worker)
    i_worker = ovl.position_in(n_worker)[0]
    i_start = ovl.cast(i_worker * n_work_per_worker, dtype=ovl.int64)
    i_end = ovl.cast(ovl.minimum((i_worker+1) * n_work_per_worker, capacity+1), dtype=ovl.int64)
    max_out = ovl.output_like(max_in)

    # For values at capacities smaller than weight carry over from the input.
    with ovl.if_(i_start < weight):
        for i in ovl.arange(i_start, ovl.minimum(i_end, weight)):
            max_out[i] = max_in[i]
            i_start = weight

    # For all remaining capacities compute the maximum of the trailing value + value and the current value.
    for i in ovl.arange(i_start, i_end+1):
        v = max_in[i-weight] + value
        with ovl.if_(max_in[i] < v):
            max_out[i] = v
        with ovl.else_():
            max_out[i] = max_in[i]

    # Return the new maximum values.
    return max_out


def kp(values, weights, capacity):
    """
    Solves the Knapsack problem using dynamic programming. This implementation uses a tensorflow ops and ovl ops.
    :param values: The value of each item.
    :param weights: The weight or cost of taking an item.
    :param capacity: The capacity limitation of the knapsack in total weight or cost.
    :return: The maximum achievable value while maintaining the capacity limit.
    """
    i = tf.constant(0, dtype=np.float64)
    n = tf.constant(len(values), dtype=np.float64)

    # We stop after n iterations.
    def condForWhile(max_in, values, weights, i, n):
        return tf.less(i, n)

    # In each iteration we compute the maxima.
    def bodyForWhile(accu, values, weights, i, n):
        i2 = tf.reshape(i, [1])
        op = kp_max(accu, values, weights, i2, capacity=capacity, n_work_per_worker=50)
        accu = ovl.as_tensorflow(op)
        return [accu, values, weights, i + 1, n]

    # Initialize the value per capacity variable and while loop i < n with i = 0 initially.
    max_in = tf.convert_to_tensor(np.zeros((capacity+1), dtype=np.float64))
    max_in, values, weights, i, n = tf.while_loop(condForWhile, bodyForWhile, [max_in, values, weights, i, n])

    # Run a tensorflow session.
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        max_out = sess.run(max_in)

    return max_out[capacity]

class TestKnapsack(unittest.TestCase):
    """
    Test cases for the knapsack operator.
    """
    def test(self):
        """
        This tests the knapsack operator for 5, 10, 50, 100 items and non-trivial capacities.
        """
        n_cases = [5, 10, 50, 100]
        for n_case in n_cases:
            ovl.logger.debug("Test case n = %d." % (n_case))
            weights = np.random.random(n_case)*n_case
            capacity = int(0.5*np.sum(weights, axis=0))
            values = weights + 50
            if ovl.local.cuda_enabled:
                assert kp_py(values, weights, capacity) == kp(values, weights, capacity)

if __name__ == '__main__':
    ovl.clear_op_cache()
    unittest.main()




