
The Operator Vectorization Library, or OVL, is a python library for defining high performance
custom operators for the TensorFlow platform. OVL enables TensorFlow users
to easily write, test, and use custom operators in pure python without sacrificing performance. This circumvents the
productivity bottleneck of implementing, building, and linking custom C++ and CUDA operators or propagating
them through the Eigen code base.

Key features include:

* A *single python implementation* is used to generate both C++ and CUDA operators and transparently link them
  into the TensorFlow run time, cutting out the overhead of implementing, testing, and maintaining operators for
  multiple hardware architectures.
* An *optimizer* which can fuse an unbounded number of qualifying OVL operators into a single function call,
  mitigating performance bottlenecks related to global memory bandwidth limits and operator launch overhead.
* A *python testing framework* so that users can directly test and profile their operators,
  on both the CPU and GPU, against a python-native reference like numpy.
* Straightforward *gradient definition*, enabling OVL operators to be fully integrated in the middle of a
  broader neural network or other gradient-dependent TensorFlow graph.
* A *standard library* of OVL operators which can be optimized in conjunction with user defined ops and used for
  defining operator gradients

OVL operators are *not* intended to replace performance-tuned core API operators (e.g. CUDNN library calls). OVL
operators are *not* intended to replace all custom operator use cases -- sometimes there is no substitute
to an expert writing/tuning a C++/CUDA operator directly. For everything else, the mission of OVL
is to offer a significant productivity gain for implementing custom ops without introducing performance bottlenecks.

Users write OVL ops by implementing a vectorized function which statelessly maps input tensors into output tensors.
The key abstraction of the OVL programming model is the *parallel for*, aka *map*. The user defines an abstractly shaped
workgroup and a worker function which is applied over the indices of that workgroup. Each worker knows its
indices within the workgroup and can read (write) from (to) any point in the input (output) tensors.

An example code snippet of a slightly more than trivial OVL operator for shifting the values of a 1D tensor by 1 element:

.. testcode::

    import tensorflow as tf
    import opveclib as ovl


    @ovl.operator()
    def shift_cyclic(input_tensor):
        # make sure input is 1D
        assert input_tensor.rank == 1

        # define the output tensor
        output_tensor = ovl.output_like(input_tensor)

        # define the workgroup shape and get workgroup position reference
        wg_position = ovl.position_in(input_tensor.shape)

        # read input element at current workgroup position
        input_element = input_tensor[wg_position]

        # define the output position to be 1 greater than the input/workgroup position
        output_position = (wg_position[0] + 1) % input_tensor.size

        # set the output element
        output_tensor[output_position] = input_element

        return output_tensor


    a = tf.constant([0, 1, 2, 3, 4], dtype=tf.float32)
    b = ovl.as_tensorflow(shift_cyclic(a))

    sess = tf.Session()
    print(sess.run([b]))


Outputs:

.. testoutput::

   [array([ 4.,  0.,  1.,  2.,  3.], dtype=float32)]


Documentation
-------------
Full documentation is available `here <http://opveclib.readthedocs.io/>`__.


Installation
------------
OVL is currently tested and supported under Ubuntu 14.04, python 2.7 and 3.4, and is compatible with both the CPU and
GPU versions of TensorFlow. OVL requires a c++ compiler to be available on the system and is currently only tested with
g++. OVL has been tested with the following nvidia GPUs:

::
    GeForce GTX TITAN
    GeForce GTX TITAN Black


Install TensorFlow
~~~~~~~~~~~~~~~~~~
OVL requires TensorFlow 0.10.0 and works with both the CPU and GPU versions. Installation instructions
are available `here <https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#download-and-setup>`__.
Users are recommended follow the TensorFlow installation guide and its testing/troubleshooting recommendations
before using OVL.

OVL detects which version of TensorFlow, CPU or GPU, is installed at runtime. If the GPU version is installed, both
CPU and GPU versions of the operators will be generated. To do so, OVL requires access to CUDA which
should have been installed already during the GPU-enabled TensorFlow installation process. OVL assumes the CUDA
install path to be '/usr/local/cuda' - if this is incorrect the user must set the correct path in the 'CUDA_HOME'
environment variable.

Install c++ compiler and nose2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
OVL requires a c++ compiler to be available in order to generate operators that run on the CPU. The default c++ compiler
is g++, but this can be overridden by setting a custom compiler path in the OPVECLIB_CXX environment variable. OVL
uses nose2 to run tests, so it is recommended to install as well to test the installation.

.. code-block:: console

    sudo apt-get install python-nose2 g++


Install opveclib
~~~~~~~~~~~~~~~~

Install the latest release of OVL:

.. code-block:: console

    sudo pip install --upgrade opveclib

If you see an error message during the install like
``libcublas.so.7.5: cannot open shared object file: No such file or directory``, this likely means that the CUDA
library path is not exposed to the sudo environment. To solve this issue you
may explicitly pass an ``LD_LIBRARY_PATH`` to sudo to install the package:

.. code-block:: console

    sudo LD_LIBRARY_PATH=/usr/local/cuda/lib64 pip install --upgrade opveclib


Test your installation
~~~~~~~~~~~~~~~~~~~~~~

To test that your installation is correct, run the OVL build acceptance test:

.. code-block:: console

    nose2 -F opveclib.test -A '!regression' --verbose


Troubleshooting
~~~~~~~~~~~~~~~

The GPU version of TensorFlow requires CUDA to be installed on your system. Depending on how CUDA is installed,
you may need to explicitly set the CUDA_HOME environment variable, typically:

.. code-block:: console

    export CUDA_HOME=/usr/local/cuda


If you see an error like: ``libcublas.so.7.5: cannot open shared object file: No such file or directory``
You may also need to make sure the CUDA libraries are on your library path, typically:

.. code-block:: console

    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


Current Build Status
--------------------
TODO: Expose Jenkins status here.

