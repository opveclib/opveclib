# Copyright 2016 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import pickle
import itertools
import matplotlib.pyplot as plt
import opveclib as ops

methods = ['TF', 'OVL']
if ops.cuda_enabled:
    devices = ['CPU', 'GPU']
else:
    devices = ['CPU']

with open('/tmp/results.p', 'rb') as handle:
    results = pickle.load(handle)


color_map = {'OVL': 'b', 'TF': 'r'}
symbol_map = {'CPU': 'o', 'GPU': '^'}
label_map = {'OVL': 'OVL', 'TF': 'TensorFlow'}
plot_map = {'CPU': 411, 'GPU': 413}
for method, device in itertools.product(methods, devices):
    cur_table = results[method][device]
    plt.subplot(plot_map[device])
    plt.loglog(cur_table[:, 0], cur_table[:, 1], label=label_map[method],
               marker=symbol_map[device], color=color_map[method])
    plt.xlabel('Tensor elements')
    plt.ylabel('Execution time (ms)')

    if method not in results.keys():
        results[method] = {}
    results[method][device] = cur_table


plt.subplot(411)
plt.legend(loc='upper left', title='CPU')
plt.subplot(412)
cpu_speedup = results['TF']['CPU'][:, 1]/results['OVL']['CPU'][:, 1]
elements = results['OVL']['CPU'][:, 0]
plt.semilogx(elements, cpu_speedup, marker=symbol_map['CPU'], color='k')
plt.xlabel('Tensor elements')
plt.ylabel('CPU speedup')

if ops.cuda_enabled:
    plt.subplot(413)
    plt.legend(loc='upper left', title='GPU')
    plt.subplot(414)
    gpu_speedup = results['TF']['GPU'][:, 1]/results['OVL']['GPU'][:, 1]
    plt.semilogx(elements, gpu_speedup, marker=symbol_map['GPU'], color='k')
    plt.ylabel('GPU speedup')
    plt.xlabel('Tensor elements')
    plt.tight_layout()
plt.show()

