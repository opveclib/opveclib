# Copyright 2016 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import os
import itertools
import subprocess
import string
import pickle
import numpy as np
import matplotlib.pyplot as plt
import opveclib as ops

perf_name = 'perf_elementwise_l2.py'
methods = ['OVL', 'TF']
if ops.cuda_enabled:
    devices = ['CPU', 'GPU']
else:
    devices = ['CPU']
lengths = np.array([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
iterations = 1e5

this_file_path = os.path.abspath(__file__)
this_directory = os.path.split(this_file_path)[0]
performance_target = os.path.join(this_directory, perf_name)

results = {}
color_map = {'OVL': 'b', 'TF': 'r'}
symbol_map = {'CPU': 'o', 'GPU': '^'}
for method, device in itertools.product(methods, devices):
    cur_table = None
    for length in lengths:
        try:
            stdout = subprocess.check_output(['python',
                                              performance_target,
                                              '-d', device,
                                              '-l', str(length),
                                              '-m', method,
                                              '-i', str(iterations),
                                              '-csv'])
        # throw out results that failed
        except:
            pass
        else:
            result_list = string.split(stdout, ', ')
            result_list = [length] + result_list
            cur_results = np.array(result_list).astype(np.float32)

            if cur_table is None:
                cur_table = cur_results
            else:
                cur_table = np.row_stack((cur_table, cur_results))

    plt.loglog(cur_table[:, 0], cur_table[:, 1], label=method+'_'+device,
               marker=symbol_map[device], color=color_map[method])

    if method not in results.keys():
        results[method] = {}
    results[method][device] = cur_table

plt.legend(loc='upper left')
plt.show()

with open('/tmp/results.p', 'wb') as handle:
    pickle.dump(results, handle)
