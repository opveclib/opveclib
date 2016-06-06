# Copyright 2016 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
# the specific language governing permissions and limitations under the License.


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from clustering import createClusterData, initialClusterCenters, kMeansGPU

if __name__ == '__main__':
    """Demo program for kMeans clustering.

    This program generates a dataset of 3D points and the intial cluster centers,
    performs the clustering, and visualizes the result.
    """
    data, clusterGt = createClusterData()
    cluster = initialClusterCenters()
    clusterGPU  = kMeansGPU(data, cluster, nMaxIter=10, th=1e-4)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[0,:], data[1,:], data[2,:],c='b',marker='.',s=5)
    ax.scatter(clusterGPU[0,:],clusterGPU[1,:],clusterGPU[2,:],c='r',marker='s',s=80)
    plt.show()