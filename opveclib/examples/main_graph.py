# Copyright 2016 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
# the specific language governing permissions and limitations under the License.

from __future__ import absolute_import
import os.path
from test_graph import loadGraphFromTextFile, countTrianglesCPU, countTrianglesGPU, countTrianglesNp
import opveclib as ovl
from six.moves import urllib
import gzip
import os.path

def downloadAndUnzipGraph(urlName, tmpName):
    """Download, unzip, and write a graph to a file.

    :param urlName: URL of the data to download.
    :type urlName: String.
    :param tmpName: Name of graph in temporary folder.
    :type tmpName: String.
    """

    # Cache the downloaded file in the /tmp directory and only download it again if not present.
    if not os.path.isfile(tmpName):
        print("Downloading data file %s." % (urlName))

    f = urllib.request.urlopen(urlName)
    tmpNameGz = tmpName + ".gz"
    with open(tmpNameGz, 'wb') as fCompressed:
        fCompressed.write(f.read())
        fCompressed.close()
        decompressedFile = gzip.GzipFile(tmpNameGz, mode='rb')
        with open(tmpName, 'w') as fDecompressed:
            fDecompressed.write(decompressedFile.read().decode("utf-8"))

if __name__ == '__main__':
    """Demo program for the counting of triangles in an undirected graph.

    This program loads an undirected graph and counts its triangles.

    """

    urlName     = "https://snap.stanford.edu/data/ca-HepTh.txt.gz"
    tmpName     = "/tmp/ca-HepPh.txt"

    downloadAndUnzipGraph(urlName, tmpName)

    if os.path.isfile(tmpName):
        startEdge, fromVertex, toVertex = loadGraphFromTextFile(tmpName)

        nTriangleNPY = countTrianglesNp(startEdge, fromVertex, toVertex)
        nTriangleCPU = countTrianglesCPU(startEdge, fromVertex, toVertex)

        # Print the file name.
        print('Counting triangles for graph %s.' % (urlName))
        # Print the count.
        print('Counted %d triangles using numpy.' % (nTriangleNPY))
        print('Counted %d triangles using OVL + CPU.' % (nTriangleCPU))

        if ovl.local.cuda_enabled:
            nTriangleGPU = countTrianglesGPU(startEdge, fromVertex, toVertex)
            print('Counted %d triangles using OVL + GPU.' % (nTriangleGPU))
