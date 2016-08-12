# Copyright 2016 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import numpy as np
import opveclib as ovl
import tensorflow as tf
import itertools
import time


@ovl.operator()
def conv_1d(x, v, kernel_orientation='as-is', stride=1, mode='same', data_format='NCE'):
    """
    Define the operator function.

    :param x: An input tensor of shape [num_batches, num_channels, num_elements].
    :param v: A filter/kernel of shape [num_filters, num_channels, kernel_size].
    :param kernel_orientation: The orientation of the kernel to use: 'as-is' or 'flipped'. This language is used
        rather than 'convolution' or 'cross-correlation' since the terms have become overloaded and ambiguous across
        some fields. As defined in https://en.wikipedia.org/wiki/Cross-correlation#Properties, 'as-is' yields the
        cross-correlation and 'flipped' yields the convolution.
    :param stride: kernel stride to use.
    :param mode: border mode: 'same', 'valid', or 'full'
    :param data_format: order of the dimensions in the input: 'NCE', 'NEC' etc.
    :return: an output tensor of shape [num_batches, num_filters, num_elements]
    """

    if kernel_orientation != 'as-is' and kernel_orientation != 'flipped':
        raise ValueError("kernel_orientation must be 'as-is' or 'flipped'")

    # resolve data layout based on data_format input
    assert x.rank == 3
    assert len(data_format) == 3
    assert data_format.count('N') == 1
    assert data_format.count('C') == 1
    assert data_format.count('E') == 1

    n_axis = data_format.find('N')
    c_axis = data_format.find('C')
    e_axis = data_format.find('E')

    num_elements = x.shape[e_axis]
    num_channels = x.shape[c_axis]
    num_batches = x.shape[n_axis]

    assert v.rank == 3
    if num_channels != v.shape[c_axis]:
        raise ValueError('Channel axis size of input must match that of the filter.')

    num_filters = v.shape[n_axis]
    filter_size = v.shape[e_axis]
    left_apron = filter_size // 2
    right_apron = filter_size - left_apron - 1

    if not isinstance(stride, int) or stride < 1 or stride > num_elements:
        raise ValueError('Stride must be a positive integer')

    if mode == 'same':
        if filter_size > num_elements:
            raise ValueError('filter size, ' + str(filter_size) +
                             ',  cannot be larger than number of elements, ' + str(num_elements))

        starting_element = -left_apron
        ending_element = num_elements - left_apron
    elif mode == 'valid':
        if filter_size > num_elements:
            raise ValueError('filter size, ' + str(filter_size) +
                             ',  cannot be larger than number of elements, ' + str(num_elements))

        starting_element = 0
        ending_element = num_elements - (left_apron + right_apron)
    elif mode == 'full':
        starting_element = -(filter_size - 1)
        ending_element = num_elements
    else:
        raise ValueError("mode must be 'same', 'valid', or 'full'.")

    output_elements = (ending_element - starting_element)

    output_shape = [0, 0, 0]
    output_shape[n_axis] = num_batches
    output_shape[c_axis] = num_filters
    output_shape[e_axis] = output_elements
    output = ovl.output(output_shape, x.dtype)

    filters_per_worker = 1
    filter_workers, filter_remainder = divmod(num_filters, filters_per_worker)
    if filter_remainder > 0:
        filter_workers += 1

    batches_per_worker = 1
    batch_workers, batch_remainder = divmod(num_batches, batches_per_worker)
    if batch_remainder > 0:
        batch_workers += 1

    elements_per_worker = 10
    element_workers, element_remainder = divmod(output_elements, elements_per_worker)
    if element_remainder > 0:
        element_workers += 1

    workgroup_shape = [batch_workers, filter_workers, element_workers]
    ovl.logger.debug(u'    workgroup_shape: ' + str(workgroup_shape))
    pos = ovl.position_in(workgroup_shape)
    cur_batch_block = pos[0]
    cur_filter_block = pos[1]
    cur_element_block = pos[2]

    num_block_batches = ovl.variable(batches_per_worker, ovl.uint32)
    if batch_remainder > 0:
        with ovl.if_(cur_batch_block == batch_workers-1):
            num_block_batches <<= batch_remainder

    num_block_filters = ovl.variable(filters_per_worker, ovl.uint32)
    if filter_remainder > 0:
        with ovl.if_(cur_filter_block == filter_workers-1):
            num_block_filters <<= filter_remainder

    num_block_elements = ovl.variable(elements_per_worker, ovl.uint32)
    if element_remainder > 0:
        with ovl.if_(cur_element_block == element_workers-1):
            num_block_elements <<= element_remainder

    accum = ovl.zeros((batches_per_worker, filters_per_worker, elements_per_worker), ovl.float64) #4*4

    filter_block = ovl.zeros((filters_per_worker, filter_size), v.dtype)  #4*10
    input_block = ovl.zeros((batches_per_worker, filter_size), x.dtype)  #4*10
    for cur_channel in ovl.arange(num_channels):

        # load all filters for this channel
        for intra_block_filter in ovl.arange(filters_per_worker):
            for f_pos in ovl.arange(filter_size):
                filter_index = [None, None, None]
                filter_index[c_axis] = cur_channel
                filter_index[n_axis] = ovl.cast(intra_block_filter, ovl.uint32) + cur_filter_block * filters_per_worker
                if kernel_orientation == 'as-is':
                    filter_index[e_axis] = f_pos
                elif kernel_orientation == 'flipped':
                    filter_index[e_axis] = filter_size - f_pos - 1
                else:
                    raise ValueError("kernel_orientation must be 'as-is' or 'flipped'")
                filter_block[intra_block_filter, f_pos] = v[filter_index]

        # load initial inputs for this channel
        buffer_head = ovl.variable(0, ovl.uint32)
        for intra_block_batch in ovl.arange(num_block_batches):
            cur_batch = intra_block_batch + cur_batch_block*batches_per_worker
            for f_pos in ovl.arange(filter_size):
                x_index = [None, None, None]
                x_index[c_axis] = cur_channel
                x_index[n_axis] = cur_batch

                x_elem_index = starting_element + ovl.cast(cur_element_block * elements_per_worker, ovl.uint64) + ovl.cast(f_pos, ovl.uint64)
                x_index[e_axis] = x_elem_index
                index_in_bounds = ovl.logical_and(x_elem_index >= 0, x_elem_index < num_elements)
                with ovl.if_(index_in_bounds):
                    input_block[intra_block_batch, f_pos] = x[x_index]
                with ovl.else_():
                    input_block[intra_block_batch, f_pos] = 0

        for intra_block_element in ovl.arange(num_block_elements):
            cur_elem = intra_block_element + cur_element_block*elements_per_worker
            for intra_block_batch in ovl.arange(num_block_batches):
                cur_batch = intra_block_batch + cur_batch_block*batches_per_worker
                for intra_block_filter in ovl.arange(num_block_filters):
                    for f_pos in ovl.arange(filter_size):
                        x_pos = (buffer_head + ovl.cast(f_pos, ovl.uint32)) % filter_size
                        cur_x = ovl.cast(input_block[intra_block_batch, x_pos], ovl.float64)
                        cur_v = ovl.cast(filter_block[intra_block_filter, f_pos], ovl.float64)
                        accum[intra_block_batch, intra_block_filter, intra_block_element] = \
                            accum[intra_block_batch, intra_block_filter, intra_block_element] + cur_x * cur_v

                # load new element
                x_index = [None, None, None]
                x_index[c_axis] = cur_channel
                x_index[n_axis] = cur_batch
                x_elem_index = starting_element + cur_elem + filter_size
                x_index[e_axis] = x_elem_index
                index_in_bounds = ovl.logical_and(x_elem_index >= 0, x_elem_index < num_elements)
                with ovl.if_(index_in_bounds):
                    input_block[intra_block_batch, buffer_head] = x[x_index]
                with ovl.else_():
                    input_block[intra_block_batch, buffer_head] = 0

            buffer_head <<= (buffer_head + 1) % filter_size

    for intra_block_batch in ovl.arange(num_block_batches):
        cur_batch = intra_block_batch + cur_batch_block*batches_per_worker
        for intra_block_filter in ovl.arange(num_block_filters):
            cur_filter = intra_block_filter + cur_filter_block*filters_per_worker
            for intra_block_element in ovl.arange(num_block_elements):
                cur_elem = intra_block_element + cur_element_block*elements_per_worker

                output_index = [None, None, None]
                output_index[n_axis] = cur_batch
                output_index[e_axis] = cur_elem
                output_index[c_axis] = cur_filter
                output[output_index] = ovl.cast(accum[intra_block_batch, intra_block_filter, intra_block_element],
                                                output.dtype)

    return output


def reference(x, v, mode, orientation, data_format):
        # resolve data layout based on data_format input
        assert len(x.shape) == 3
        assert len(data_format) == 3
        assert data_format.count('N') == 1
        assert data_format.count('C') == 1
        assert data_format.count('E') == 1

        n_axis = data_format.find('N')
        c_axis = data_format.find('C')
        e_axis = data_format.find('E')

        num_channels = x.shape[c_axis]
        num_batches = x.shape[n_axis]
        num_elements = x.shape[e_axis]

        assert len(v.shape) == 3
        if num_channels != v.shape[c_axis]:
            raise ValueError('Channel axis size ' + str(num_channels) +
                             ' of input must match that of the filter - ' +
                             str(v.shape[c_axis]))

        num_filters = v.shape[n_axis]
        filter_size = v.shape[e_axis]
        left_apron = filter_size // 2
        right_apron = filter_size - left_apron - 1

        output_shape = [None, None, None]
        output_shape[n_axis] = num_batches
        output_shape[c_axis] = num_filters
        if mode == 'same':
            output_elements = num_elements
        elif mode == 'valid':
            output_elements = num_elements - left_apron - right_apron
        elif mode == 'full':
            output_elements = num_elements + left_apron + right_apron
        else:
            raise ValueError
        output_shape[e_axis] = output_elements

        output = np.empty(output_shape, dtype=float)

        for cur_batch in range(num_batches):
            for cur_filter in range(num_filters):
                accum = np.zeros(output_elements)
                for cur_channel in range(num_channels):
                    x_index = [None, None, None]
                    x_index[n_axis] = cur_batch
                    x_index[c_axis] = cur_channel
                    x_index[e_axis] = slice(num_elements)

                    v_index = [None, None, None]
                    v_index[n_axis] = cur_filter
                    v_index[c_axis] = cur_channel
                    v_index[e_axis] = slice(filter_size)
                    if orientation == 'as-is':
                        accum += np.correlate(x[x_index], v[v_index], mode=mode)
                    elif orientation == 'flipped':
                        accum += np.convolve(x[x_index], v[v_index], mode=mode)
                    else:
                        raise RuntimeError()
                output_index = [None, None, None]
                output_index[n_axis] = cur_batch
                output_index[c_axis] = cur_filter
                output_index[e_axis] = slice(output_elements)
                output[output_index] = accum

        return output

def run_tf(tensor_in_sizes, filter_in_sizes):
    # test TF  2D convolution operator in 1D vs. OVL
    total_size_1 = 1
    total_size_2 = 1
    for s in tensor_in_sizes:
      total_size_1 *= s
    for s in filter_in_sizes:
      total_size_2 *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x1 = [f * 1.0 for f in range(1, total_size_1 + 1)]
    x2 = [f * 1.0 for f in range(1, total_size_2 + 1)]
    tin1 = tf.constant(x1, shape=tensor_in_sizes, dtype=tf.float32)
    tin2 = tf.constant(x2, shape=filter_in_sizes, dtype=tf.float32)
    conv = tf.nn.conv2d(tin1, tin2,
                      strides=[1, 1, 1, 1],
                      padding="SAME",
                      data_format='NHWC')

    # compare to OVL - need to convert input to 1-D - ie. input_rows = filter_rows = 1
    # also transpose initial data since filter index is last in TF and first in OVL
    # TF input = batch, input_row, input_col, channels
    # TF filter = filter_row, filter_col, channels, num_filters
    # OVL NEC input = batches, num_elements, channels
    # OVL NEC filter = num_filters, kernel_size, channels
    assert(tensor_in_sizes[1] == 1)
    assert(filter_in_sizes[0] == 1)
    ovl_tensor_in_sizes = [tensor_in_sizes[0], tensor_in_sizes[2], tensor_in_sizes[3]]
    num_filter = filter_in_sizes[3]
    num_elem = filter_in_sizes[1]
    num_chan = filter_in_sizes[2]
    ovl_filter_in_sizes = [num_filter, num_elem, num_chan]
    ovl.logger.debug(u'input and filter sizes: ' + str(ovl_tensor_in_sizes) + ', ' + str(ovl_filter_in_sizes))
    ar1 = np.array(x1, dtype=np.float).reshape(ovl_tensor_in_sizes)
    # does not produce the correct results
    # ar2 = np.array(x2, dtype=np.float).reshape(ovl_filter_in_sizes, order='F')
    ar2 = np.zeros(ovl_filter_in_sizes, dtype=np.float)
    for col in range(0, num_elem):
        for chan in range(0, num_chan):
            for num in range(0, num_filter):
                index = col * num_chan * num_filter + chan * num_filter + num
                # print('ar2 ' + str(num) + ',' + str(col) + ',' + str(chan) + ' is index ' + str(index) + ' val: ' + str(x2[index]))
                ar2[num,col,chan] = x2[index]

    t0 = time.time()
    ref = reference(ar1, ar2, mode='same', orientation='as-is', data_format= 'NEC')
    t1 = time.time()
    np_time = (t1-t0)*1000

    iters = 100
    ovlOp = conv_1d(ar1, ar2, mode='same', kernel_orientation='as-is', data_format='NEC')
    if ovl.cuda_enabled:
        ovlResult, prof = ovl.profile(ovlOp, target_language='cuda', profiling_iterations=iters, opt_level=3)
        ovl_cuda_time = np.min(list(prof.values())[0])
        assert np.allclose(ovlResult, ref)
    #TODO - cpp is really slow...
    ovlcppResult, profcpp = ovl.profile(ovlOp, target_language='cpp', profiling_iterations=iters, opt_level=3)
    ovl_cpp_time = np.min(list(profcpp.values())[0])
    assert np.allclose(ovlcppResult, ref)

    # ensure TF runs on GPU
    test_config=tf.ConfigProto(allow_soft_placement=False)
    test_config.graph_options.optimizer_options.opt_level = -1

    # OVL-TF integration
    ovl_tf_time = 0
    with tf.Session(config=test_config) as sess:
        with tf.device('/gpu:0'):
            ovlOp_tf = ovl.as_tensorflow(ovlOp)
            init = tf.initialize_all_variables()
            sess.run(init)
            ovlOp_tf_result = sess.run(ovlOp_tf)
            t0 = time.time()
            for dummy in itertools.repeat(None, iters):
                sess.run(ovlOp_tf.op)
            t1 = time.time()
            ovl_tf_time = (t1-t0)/float(iters) * 1000.00
            assert np.allclose(ovlOp_tf_result, ref)
    sess.close()

    # run TF 2D conv alone
    tf_time = 0
    with tf.Session(config=test_config) as sess:
        with tf.device('/gpu:0'):
            result = sess.run([conv])
            t0 = time.time()
            for dummy in itertools.repeat(None, iters):
                sess.run([conv.op])
            t1 = time.time()
            tf_time = (t1-t0)/float(iters) * 1000.00
            # TF result is 4D - have to convert to 3D to match OVL
            tf_shape = result[0].shape
            assert(tf_shape[1] == 1)
            ovl_shape = [tf_shape[0], tf_shape[2], tf_shape[3]]
            tf_result = np.array(result[0], dtype=np.float).reshape(ovl_shape)
            #TODO - if number of filter elements is even, TF result does not match reference - first element "wraps" to end
            assert np.allclose(tf_result, ref)
    sess.close()
    times = [np_time, ovl_cuda_time, ovl_cpp_time, ovl_tf_time, tf_time]
    ovl.logger.debug(u'    time [np, OVL_cuda, OVL_cpp, OVL_TF, TF]: ' + str(times))


def run_tests():
    bb = 1
    cc = 1
    ee = 1000
    k_num = 10
    a1 = np.random.random((bb, ee, cc))
    a2 = np.random.random((bb, ee, cc))
    for k_ee in range(13, 14):
        b = np.random.random((k_num, k_ee, cc))
        for md in ['valid', 'same', 'full']:
            for orientation in ['as-is', 'flipped']:
                import time
                t1 = time.time()
                y1 = reference(a1, b, md, orientation, 'NEC')
                t2 = time.time()
                y2 = reference(a2, b, md, orientation, 'NEC')
                op = conv_1d(a1, b, mode=md, kernel_orientation=orientation, data_format='NEC')

                # result1 =
                assert np.allclose(ovl.evaluate(op, target_language='cuda'), y1)
                # for d in range(1):
                a1[:] = a2[:]
                assert np.allclose(ovl.evaluate(op, target_language='cuda'), y2)

                res, prof = ovl.profile(op, target_language='cuda', profiling_iterations=100, opt_level=3)

                # print(prof)
                # print(debug)
                # print(op[0, 0, :])
                ovl.logger.debug(k_ee, md, orientation, (t2 - t1) * 1000, np.min(list(prof.values())[0]))
                # assert np.allclose(result1, y1)
                # assert np.allclose(result2, y2)


#TODO - OVL evaluate fails if it is run after a TF session
run_tf([5, 1, 1000, 3], [1, 13, 3, 10])
# run_tests()

# op = Convolution1D(np.reshape(a, (batches, chans, elems)), np.reshape(v, (chans, kern_elems)))

    # @staticmethod
    # def _conv_core(input, filter, n_axis, c_axis, e_axis, kernel_orientation, border_policy,
    #                upsample_factor=None, downsample_factor=None):
    #     filters_per_worker = 3
    #     batches_per_worker = 5
    #     strides_per_worker = 100
    #
    #     def mod_ceil(x, y):
    #         m, remainder = divmod(x, y)
    #         if remainder == 0:
    #             return m
    #         else:
    #             return m + 1
    #
    #     filter_workers = mod_ceil(num_filters, filters_per_worker)
    #     batch_workers = mod_ceil(num_batches, batches_per_worker)
    #
    #

        # element_workers, final_worker_elements = divmod(elements, elements_per_worker)
        # if final_worker_elements != 0:
        #     element_workers += 1

        # batches_per_worker = 1
        # batch_workers, final_worker_batches = divmod(batches, batches_per_worker)
        # if final_worker_batches != 0:
        #     batch_workers += 1
        #
        # workgroup_shape = [batches_per_worker, channels, element_workers]
        # workgroup_position = ops.position_in(workgroup_shape)
        #
        # cur_batch_worker = workgroup_position[0]
        # cur_channel = workgroup_position[1]
        # cur_element_worker = workgroup_position[2]
        #
        # for cur_batch in ops.arange(cur_batch_worker*batches_per_worker, (cur_batch_worker+1)*batches_per_worker):
        #     for cur_element in ops.arange(cur_element_worker*elements_per_worker,
        #                                   (cur_element_worker+1)*elements_per_worker):
        #         with ops.if_(ops.logical_and(cur_batch < batches, cur_element < elements)):
        #             pass
