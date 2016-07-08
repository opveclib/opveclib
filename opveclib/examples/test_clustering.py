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
from sys import _getframe
import numpy as np
import opveclib as ops
import sys
import tensorflow as tf

class KMeansMinDistOp(ops.Operator):
    """Minimum distance computation for kMean clustering.

    For each data point find the closest cluster center and keep that assignment stored in an index.
    """
    def op(self, data, center):
        """The definition of the operator function.

        Thread pool over nData. This may generate too many threads for some input data.

        :param data: 2D matrix as data input with dimensions: nDim x nData.
        :type data; numpy array.
        :param center: 2D matrix of initial cluster centers with dimensions: nDim x nCenter.
        :type center: numpy array.
        :return: a 1D matrix of assignemnts of data points to cluster centers: nData x 1.
        """
        nDimData    = data.shape[0]
        nDimCenter  = center.shape[0]
        nData       = data.shape[1]
        nCenter     = center.shape[1]
        assert nDimData == nDimCenter, "Data has % dimensions and centers have %d dimensions, but these must match!" % (nDimData, nDimCenter)
        nDim        = nDimData
        iSample     = ops.position_in(nData)[0]
        minDist     = ops.variable(sys.float_info.max, data.dtype)
        iMin        = ops.variable(0, ops.int64)
        for iCenter in ops.arange(nCenter):
            dist = ops.variable(0, center.dtype)
            for iDim in ops.arange(nDim):
                dist <<= dist + (data[iDim,iSample]-center[iDim,iCenter])*(data[iDim,iSample]-center[iDim,iCenter])
            with ops.if_(dist < minDist):
                iMin <<= iCenter
                minDist <<= dist

        #TODO(raudies@hpe.com): Change this to uint64 whenever ovl supports non-floating point types for tensors.
        minIndex = ops.output(nData, ops.float64) # Use float64 because tensorflow does not support uint64 as type yet.
        minIndex[iSample] = ops.cast(iMin, ops.float64)

        return minIndex

class KMeansNewCenOp(ops.Operator):
    """Computation of new cluster centers for kMeans.

    Use the assignment index of closest data points for a cluster center and recomputes the cluster centers
    as the centroid of that assigned data.
    """
    def op(self, data, minIndex, nCenter):
        """The definition of the operator function.

        Thread pool over nCenter. That should be fine for most cases.

        :param data: 2D matrix as data input with dimensions: nDim x nData.
        :type data: numpy array.
        :param minIndex: 1D matrix of assignemnts of data points to cluster centers: nData x 1.
        :type minIndex: numpy array.
        :return: a 2D matrix with computed cluster centers with dimensions> nDim x nCenter.
        """
        nDim    = data.shape[0]
        nData   = data.shape[1]
        assert nData==minIndex.shape[0], "Data has %d values and minDist has %d values, but these must match!" % (nData, minIndex.shape[0])
        iCenter = ops.position_in(nCenter)[0]
        center  = ops.zeros([nDim,nCenter], data.dtype)
        count   = ops.variable(0, data.dtype)
        for iSample in ops.arange(nData):
            with ops.if_(iCenter==ops.cast(minIndex[iSample], ops.uint32)):
                count <<= count + 1
                for iDim in ops.arange(nDim):
                    center[iDim,iCenter] = center[iDim,iCenter] + data[iDim,iSample]

        newCenter = ops.output([nDim,nCenter],data.dtype)
        for iDim in ops.arange(nDim):
            newCenter[iDim,iCenter] = center[iDim,iCenter]/count
        return newCenter

def condForWhile(iter, nMaxIter, rms, th, data, center):
    return tf.logical_and(tf.less(iter, nMaxIter), tf.less(th, rms))

def bodyForWhile(iter, nMaxIter, rms, th, data, center):
    oldCenter = center
    minIndex = KMeansMinDistOp(data, center)
    center = ops.as_tensorflow(KMeansNewCenOp(data, minIndex, nCenter=int(center.get_shape()[1])))
    rms = tf.reduce_sum(tf.sqrt(tf.reduce_sum((center-oldCenter)*(center-oldCenter), 0)), 0)
    return [iter+1, nMaxIter, rms, th, data, center]

def kMeansGPU(data, center, nMaxIter, th):
    """Clustering data using the kMeans method implemented with OVL operators and tensorflow.

    :param data: 2D matrix as data input with dimensions: nDim x nData.
    :type data: numpy array.
    :param center: 2D matrix with initial cluster centers with dimensions: nDim x nCenter.
    :type center: numpy array.
    :param nMaxIter: Maximum number of iterations.
    :type nMaxIter: int.
    :param th: Threshold applied to RMS error between prior and current cluster centers.
    :type th: float.
    :return: a 2D matrix with computed cluster centers with dimensions> nDim x nCenter.

    :Description:
        We assume that the initial cluster centers or centroids are given by the user.

        The kmeans algorithm iterates over the two steps until convergence or a maximum number of steps is reached:

        1. Find for each data point the closest cluster center and keep that assignment stored in an index.

        2. Use the assignment index and recompute the cluster centers as the centroid of the assigned data.

    :Examples:

    .. doctest::

        >>> from opveclib.examples.test_clustering import createClusterData, initialClusterCenters, kMeansGPU
        >>> data, clusterGt = createClusterData()
        >>> cluster = initialClusterCenters()
        >>> clusterGPU  = kMeansGPU(data, cluster, nMaxIter=10, th=1e-4)
    """
    # Initialize the variables as tensorflow variables.
    iter        = tf.constant(0)
    nMaxIter    = tf.constant(nMaxIter)
    data        = tf.convert_to_tensor(data)
    center      = tf.convert_to_tensor(center)
    rms         = tf.constant(2*th,dtype=data.dtype)
    th          = tf.constant(th,dtype=data.dtype)
    # Define the while loop with condition, body, and variables.
    iter, nMaxIter, rms, th, data, center = tf.while_loop(condForWhile, bodyForWhile, [iter, nMaxIter, rms, th, data, center])
    # Run the session.
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        return sess.run(center)

def kMeansTF(data, center, nMaxIter, th): # data: nDim x nData, center: nDim x  nCenter
    """Clustering data using the kMeans method implemented with tensorflow.

    :param data: 2D matrix as data input with dimensions: nDim x nData.
    :type data: numpy array.
    :param center: 2D matrix with initial cluster centers with dimensions: nDim x nCenter.
    :type center: numpy array.
    :param nMaxIter: Maximum number of iterations.
    :type nMaxIter: int.
    :param th: Threshold applied to RMS error between prior and current cluster centers.
    :type th: float.
    :return: a 2D matrix with computed cluster centers with dimensions> nDim x nCenter.

    :Examples:

    .. doctest::

        >>> from opveclib.examples.test_clustering import createClusterData, initialClusterCenters, kMeansTF
        >>> data, clusterGt = createClusterData()
        >>> cluster = initialClusterCenters()
        >>> clusterGPU  = kMeansTF(data, cluster, nMaxIter=10, th=1e-4)
    """
    nData   = data.shape[1]
    nCenter = center.shape[1]
    center  = tf.Variable(center)

    # Replicate data to have the dimensions: nDim x nData x nCenter
    rData       = tf.tile(tf.expand_dims(data,-1),[1, 1, nCenter]) # replicate for nCenter
    rCenter     = tf.transpose(tf.tile(tf.expand_dims(center,-1),[1, 1, nData]),perm=[0, 2, 1]) # replicate for nData

    # Get the cluster center of minimum distance for each data point.
    ssq         = tf.reduce_sum(tf.square(rData - rCenter), 0, keep_dims=True) # over nDim
    index       = tf.squeeze(tf.argmin(ssq, 2)) # min index over nCenter and remove leading dimension

    # Compute the new cluster centers based on the closest data points.
    newSum      = tf.unsorted_segment_sum(tf.transpose(data,[1,0]), index, nCenter)
    count       = tf.unsorted_segment_sum(tf.transpose(tf.ones_like(data),[1,0]), index, nCenter)
    newCenter   = tf.transpose(newSum / count,[1,0])

    # Compute the differences between the new and old cluster centers and threshold them.
    rms             = tf.reduce_sum(tf.sqrt(tf.reduce_sum((center-newCenter)*(center-newCenter), 0)), 0)
    changeCenter    = rms > th

    # Update the cluster centers if they have changed by more than the threshold value.
    with tf.control_dependencies([changeCenter]):
        doUpdates = center.assign(newCenter)

    # Initialize the tensor variables.
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    # As long as there are enough changes in the cluster centers and we have not reached the maximum number of
    # iterations, repeat the steps from above.
    changed = True
    iter    = 0
    while changed and iter < nMaxIter:
        iter += 1
        [changed, _] = sess.run([changeCenter, doUpdates])

    return sess.run(center)


def createClusterData(nDataPerCluster = 100):
    """Creates the cluster data.

    :param nDataPerCluster: Number of data samples per cluster.
    :type nDataPerCluster: int.
    :return: data with dimensions 3 x (8xnDataPerCluster) and clusterCenter with dimensions 3 x 8.
    """
    clusterCenter = np.zeros((3, 8), dtype=np.float64)
    clusterCenter[:,0] = [-3, +5, -2]
    clusterCenter[:,1] = [-1, -2, -1]
    clusterCenter[:,2] = [-2, -2, +1]
    clusterCenter[:,3] = [-2, +3, +3]
    clusterCenter[:,4] = [+1, -4, +1]
    clusterCenter[:,5] = [+2, -6, -3]
    clusterCenter[:,6] = [+2, +3, +1]
    clusterCenter[:,7] = [+5, +5, -2]
    clusterSpread = np.zeros((3, 8), dtype=np.float64)
    clusterSpread[:,0] = [0.1, 0.5, 0.2]
    clusterSpread[:,1] = [0.5, 0.3, 0.2]
    clusterSpread[:,2] = [0.5, 0.2, 0.1]
    clusterSpread[:,3] = [0.2, 0.4, 0.3]
    clusterSpread[:,4] = [0.2, 0.7, 0.4]
    clusterSpread[:,5] = [0.1, 0.1, 0.2]
    clusterSpread[:,6] = [0.3, 0.2, 0.1]
    clusterSpread[:,7] = [0.5, 0.3, 0.3]
    nDim = clusterCenter.shape[0]
    nCluster = clusterCenter.shape[1]
    nData = nCluster * nDataPerCluster
    data = np.zeros((nDim, nData), dtype=np.float64)
    for iCluster in range(nCluster):
        for iDim in range(nDim):
            data[iDim,iCluster*nDataPerCluster:(iCluster+1)*nDataPerCluster] = clusterCenter[iDim,iCluster] \
                                                                               + np.random.randn(1,nDataPerCluster)\
                                                                                 *clusterSpread[iDim,iCluster]

    return data, clusterCenter

def initialClusterCenters():
    """Define the initial cluster centers.

    :return: a 2D array with cluster center with dimension: nDim x nCluster (here 3 x 8).
    """
    clusterCenter = np.zeros((3, 8), dtype=np.float64)
    clusterCenter[:,0] = [-5, -5, -5]
    clusterCenter[:,1] = [+5, -5, -5]
    clusterCenter[:,2] = [-5, +5, -5]
    clusterCenter[:,3] = [+5, +5, -5]
    clusterCenter[:,4] = [-5, -5, +5]
    clusterCenter[:,5] = [+5, -5, +5]
    clusterCenter[:,6] = [-5, +5, +5]
    clusterCenter[:,7] = [+5, +5, +5]
    return clusterCenter

def compareClusterCenters(clusterCenterGt, clusterCenterEst, rtol=1.e-5, atol=1.e-8):
    """Compare compare cluster centers from ground-truth with estimated ones.

    :param clusterCenterGt: Ground-truth cluster centers as 2D matrix with dimensions: nDim x nClusterGt.
    :type clusterCenterGt: float.
    :param clusterCenterEst: Estimated cluster centers as 2D matrix with dimensions: nDim x nClusterEst.
    :type clusterCenterEst: float.
    :param rtol: Maximum threshold for relative error.
    :type rtol: float.
    :param atol: Maximum threshold for absolute error.
    :type atol: float.
    :return: true (or 1) if each estimated cluster center can be matched with at least one ground-truth cluster center.
    Otherwise this method returns false (or 0).
    """
    assert clusterCenterGt.shape[0]==clusterCenterEst.shape[0], "Dimensionality of cluster centers must match."
    nCenterGt = clusterCenterGt.shape[1]
    nCenterEst = clusterCenterEst.shape[1]
    # This method allows multiple estimates be matched to same ground-truth.
    # It also allows some ground-truth values not be matched by any estimate.
    allClose = 1
    for iCenterEst in range(nCenterEst):
        centerEst = clusterCenterEst[:,iCenterEst]
        anyClose = 0
        for iCenterGt in range(nCenterGt):
            centerGt = clusterCenterGt[:,iCenterGt]
            absErr = np.absolute(centerGt-centerEst)
            relErr = np.absolute(absErr/centerGt)
            anyClose |= all(e < atol for e in absErr) and all(e < rtol for e in relErr)
        allClose &= anyClose # All estimates need to be close to at least one ground-truth.
    return allClose


class TestKMeans(unittest.TestCase):
    """
    Test cases for kMeans clustering.
    """
    def test(self):
        """
        This test case generates random data with eight cluster centers and 100 data points per cluster center. Each
        datum has three components.
        """
        print('*** Running Test: ' + self.__class__.__name__ + ' function: ' + _getframe().f_code.co_name)
        np.random.seed(3)

        data, clusterCenterGt = createClusterData()
        clusterCenter = initialClusterCenters()

        # CUDA is not required to run these test cases.
        # TF will automatically run on the CPU version if that is all that is available
        clusterCenterOVL    = kMeansGPU(data, clusterCenter, nMaxIter=500, th=1e-4)
        clusterCenterTF     = kMeansTF(data, clusterCenter, nMaxIter=500, th=1e-4)

        assert compareClusterCenters(clusterCenterGt, clusterCenterOVL, rtol=0.1, atol=0.1)
        assert compareClusterCenters(clusterCenterGt, clusterCenterTF, rtol=0.1, atol=0.1)

if __name__ == '__main__':
    ops.clear_op_cache()
    unittest.main()
