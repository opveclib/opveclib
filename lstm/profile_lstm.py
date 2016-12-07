# Copyright 2016 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
# the specific language governing permissions and limitations under the License.

# from __future__ import absolute_import
import logging
import json
import argparse
from operator import itemgetter
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import opveclib as ovl
import opveclib.stdops as ops
from lstm import lstm

parser = argparse.ArgumentParser(description='Profile the LSTM.')
parser.add_argument('--iters', type=int, help='number of iterations', default=100)
parser.add_argument('--opt_level', type=int, help='optimization level', default=3)
parser.add_argument('--size', type=str, help='size of the problem', default='small')
parser.add_argument('--use_tensorflow', action='store_true')
parser.add_argument('--use_manual', action='store_true')

args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG)

use_tensorflow = args.use_tensorflow
use_manual = args.use_manual
print(use_manual)
opt_level = args.opt_level
iters = args.iters
sze = args.size
tipe = 'both'
if sze == 'small':
    batches = 20
    vec_len = 200
elif sze == 'medium':
    batches = 20
    vec_len = 650
elif sze == 'large':
    batches = 20
    vec_len = 1500
else:
    raise RuntimeError()

forget = 0.0

# dag = _build_op_dag(new_state)
# merged = _merge_op_dag(dag.proto_dag)
with tf.Graph().as_default() as g:
    with tf.device('/gpu:0'):
        # tf.Variable(
        # tf.random_normal([2, 2], mean=0.0, stddev=1.0, dtype=tf.float32)
        concat_arg = tf.Variable(tf.random_uniform((batches, 4*vec_len), seed=1))
        c = tf.Variable(tf.random_uniform((batches, vec_len), seed=2))
        d_new_c = tf.Variable(tf.random_uniform((batches, vec_len), seed=3))
        d_new_h = tf.Variable(tf.random_uniform((batches, vec_len), seed=4))
        # concat_arg = tf.random_uniform((batches, 4*vec_len), seed=1)
        # c = tf.random_uniform((batches, vec_len), seed=2)
        # d_new_c = tf.random_uniform((batches, vec_len), seed=3)
        # d_new_h = tf.random_uniform((batches, vec_len), seed=4)

        with g.name_scope('test_scope'):
            # use_tensorflow = False
            # use_manual = False
            # opt_level = 0
            if use_tensorflow:
                i, j, f, o = tf.split(1, 4, concat_arg)
                new_c = tf.mul(c, tf.sigmoid(f + forget)) + tf.sigmoid(i) * tf.tanh(j)
                new_h = tf.tanh(new_c) * tf.sigmoid(o)
                # dnc_dcat, dnc_dc = tf.gradients([new_c], [concat_arg, c], [d_new_c])
                # dnh_dcat, dnh_dc = tf.gradients([new_h], [concat_arg, c], [d_new_h])
                # grad = [dnc_dcat+dnh_dcat, dnc_dc + dnh_dc]
                grad = tf.gradients([new_c, new_h], [concat_arg, c], [d_new_c, d_new_h])
                trace_name = 'timeline_tf.ctf.json'
            else:
                if use_manual:
                    new_c, new_h = ovl.as_tensorflow(lstm(concat_arg, c, forget_bias=0))
                    grad = tf.gradients([new_c, new_h], [concat_arg, c], [d_new_c, d_new_h])
                    trace_name = 'timeline_ovl_manual.ctf.json'
                else:
                    i, j, f, o = ops.split(concat_arg, split_dim=1, num_split=4)
                    new_c = ops.mul(c,  ops.sigmoid(f)) + ops.sigmoid(i) * ops.tanh(j)
                    new_h = ops.tanh(new_c) * ops.sigmoid(o)
                    new_c, new_h = ovl.as_tensorflow([new_c, new_h], opt_level=opt_level)

                    grad = tf.gradients([new_c, new_h], [concat_arg, c], [d_new_c, d_new_h])
                    trace_name = 'timeline_ovl_no_opt.ctf.json'


if tipe == 'fwd':
    fetches = [new_c, new_h]
elif tipe == 'grad':
    fetches = [grad[0], grad[1]]
elif tipe == 'both':
    fetches = [new_c, new_h, grad[0], grad[1]]
else:
    fetches = []


# test_config = tf.ConfigProto(allow_soft_placement=False)
# test_config = tf.ConfigProto(allow_soft_placement=True)
test_config = tf.ConfigProto(allow_soft_placement=False)
test_config.graph_options.optimizer_options.opt_level = -1
with tf.Session(config=test_config, graph=g) as sess:
        times = np.zeros(iters)
        sess.run(tf.initialize_all_variables())
        sess.run(fetches)

        min_duration = None
        min_gpu_contig = None
        min_local_contig = None

        for i in range(iters):
            sess.run(tf.initialize_all_variables())
            run_metadata = tf.RunMetadata()
            tr = sess.run(
                    fetches,
                    options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                    run_metadata=run_metadata)

            trace = timeline.Timeline(step_stats=run_metadata.step_stats)

            events = json.loads(trace.generate_chrome_trace_format())[u'traceEvents']

            compute_start_events = {}
            compute_stop_events = {}
            compute_pid = None
            scheduling_start_events = {}
            scheduling_stop_events = {}
            scheduling_pid = None
            for event in events:
                if event['ph'] == 'M' \
                        and event['args']['name'] == '/gpu:0/stream:all Compute' \
                        and compute_pid is None:
                    compute_pid = event['pid']
                if event['ph'] == 'M' \
                        and event['args']['name'] == '/job:localhost/replica:0/task:0/gpu:0 Compute' \
                        and scheduling_pid is None:
                    scheduling_pid = event['pid']

                if event['pid'] == compute_pid:
                    cur_name = event[u'args'][u'name']
                    if cur_name.startswith(u'test_scope') and event[u'ph'] == u'X':
                        compute_start_events[cur_name] = event[u'ts']
                        compute_stop_events[cur_name] = event[u'ts'] + event[u'dur']

                if event['pid'] == scheduling_pid:
                    if u'args' in event.keys():
                        cur_name = event[u'args'][u'name']
                    else:
                        cur_name = u''
                    if cur_name.startswith(u'test_scope') and event[u'ph'] == u'X':
                        scheduling_start_events[cur_name] = event[u'ts']
                        scheduling_stop_events[cur_name] = event[u'ts'] + event[u'dur']

            def total_duration(start_events, stop_events):
                start_iter = start_events.iteritems()

                start_name, start = start_iter.next()
                stop = stop_events[start_name]
                for start_name, cur_start in start_iter:
                    if start > cur_start:
                        start = cur_start
                    if stop < stop_events[start_name]:
                        stop = stop_events[start_name]

                return start, stop

            def contiguous_timing(start, stop):
                start_sorted = sorted(start.iteritems(), key=itemgetter(1), reverse=False)
                outer = start_sorted[0][0]

                segment_time = [stop[outer] - start[outer]]
                for inner, inner_start in start_sorted:
                    outer_stop = stop[outer]
                    inner_stop = stop[inner]

                    if inner_start <= outer_stop:
                        if outer_stop < inner_stop:
                            # overlapping - accrue the difference and change outer to inner
                            segment_time[-1] += inner_stop - outer_stop
                            outer = inner
                        else:
                            # inner subsumed by outer, do nothing
                            pass
                    else:
                        # non-overlapping - accrue new inner and change outer to inner
                        segment_time.append(inner_stop-inner_start)
                        outer = inner
                contiguous_time = 0
                for t in segment_time:
                    contiguous_time += t
                return contiguous_time

            c_start, c_stop = total_duration(compute_start_events, compute_stop_events)
            s_start, s_stop = total_duration(scheduling_start_events, scheduling_stop_events)
            c_contig = contiguous_timing(compute_start_events, compute_stop_events)
            s_contig = contiguous_timing(scheduling_start_events, scheduling_stop_events)
            t_start = min(c_start, s_start)
            t_stop = max(c_stop, s_stop)
            dur = t_stop - t_start

            if min_duration is None or dur < min_duration:
                min_duration = dur
                min_gpu_contig = c_contig
                min_local_contig = s_contig
                with open('/tmp/'+trace_name, 'w') as f:
                    f.write(trace.generate_chrome_trace_format())

                num_gpu_events = len(compute_start_events)
                num_local_events = len(scheduling_start_events)

        if use_tensorflow:
            method = 'tf'
        else:
            if use_manual:
                method = 'ovl_manual'
            else:
                if opt_level == 3:
                    method = 'ovl_opt'
                else:
                    method = 'ovl'
        csv = 'method, size, min_duration, gpu_compute_time, local_compute_time, num_gpu_events, num_local_events\n'
        csv += method + ', ' + sze + ', ' + str(min_duration) + ', ' + str(min_gpu_contig) + ', ' + \
            str(min_local_contig) + ', ' + str(num_gpu_events) + ', ' + str(num_local_events)

        print(csv)
        # print('Min duration: ' + str(min_duration))
        # print('GPU compute time: ' + str(min_gpu_contig))
        # print('GPU occupancy: ' + str(float(min_gpu_contig)/min_duration))
        # print('local time: ' + str(min_local_contig))
        # print('local occupancy: ' + str(float(min_local_contig)/min_duration))
        # print('percent down time: ' + str(1.0 - float(min_local_contig)/min_duration))
        # print('Num compute events: ' + str(len(compute_start_events)))
        # print('Num scheduling events: ' + str(len(scheduling_start_events)))
