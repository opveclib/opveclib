
The Operator Vectorization Library, or OVL, is a python productivity library for defining high performance
custom operators for the TensorFlow platform. OVL enables TensorFlow users
to easily write, test, and use custom operators in pure python without sacrificing the performance gains of
implementing, building, and linking custom C++ or CUDA operators. OVL operators are *not* intended to
replace the performance-tuned core API operators, but rather augment the API in cases
where it does not cover the end-user's application particularly well. OVL operators are *not* intended to replace
*all* custom operators -- sometimes there is no substitute to a human writing/tuning the C++/CUDA directly for
an especially complicated operator that resides in a performance-critical section. Ultimately, the mission of OVL
is to offer a 10x productivity gain for custom ops which exhibit at least 80% of the performance of an equivalently
hand-written op.

Users write OVL ops by implementing a vectorized function which statelessly maps input tensors into output tensors.
The key abstraction of the OVL programming model is a parallel *for*, or *map*. The user defines an abstractly shaped
workgroup and a worker function which is applied over the indices of that workgroup. Each worker knows its
indices within the workgroup and can read (write) from (to) any point in the input (output) tensors.

Current Build Status
--------------------
TODO: Provide link to Jenkins status here.

Installation instructions
-------------------------
OVL is currently tested and supported under Ubuntu 14.04. These instructions will prepare an Ubuntu
environment to use OVL.

Install TensorFlow
~~~~~~~~~~~~~~~~~~
OVL depends on TensorFlow 0.9.0 and works with both the CPU and GPU versions. Installation instructions
are available
`here <https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#download-and-setup>`_.

Note that the GPU version of TensorFlow requires the CUDA SDK 7.5 and cuDNN 4 to be installed on your system.
Depending on how these dependencies are installed you may need to explicitly set the CUDA_HOME environment variable,
typically:

.. code-block:: console

    export CUDA_HOME=/usr/local/cuda


If you see an error like: ``libnvrtc.so.7.5: cannot open shared object file: No such file or directory``
You may also need to make sure the CUDA libraries are on your library path, typically:

.. code-block:: console

    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


Install the Operator Vectorization Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, if you do not have pip >= 8.1.1 installed, install or upgrade it. You can check the installed version
by running:

.. code-block:: console

    pip --version


If it is not installed or you have a version lower than 8.1.1 installed, upgrade it:

.. code-block:: console

    curl -O https://bootstrap.pypa.io/get-pip.py
    sudo python get-pip.py

Install dependencies: nose2, protobuf compiler/libraries and g++:

.. code-block:: console

    sudo apt-get install python-nose2 protobuf-compiler libprotobuf-dev g++

Install the latest release of OVL:

.. code-block:: console

    sudo pip install git+https://github.com/hpe-cct/opveclib

If you see an error message during the install like
``libnvrtc.so.7.5: cannot open shared object file: No such file or directory`` you
may need to explicitly pass an ``LD_LIBRARY_PATH`` to sudo to install the package:

.. code-block:: console

    sudo LD_LIBRARY_PATH=/usr/local/cuda/lib64 pip install git+https://github.com/hewlettpackardlabs/opveclib


Test your installation
~~~~~~~~~~~~~~~~~~~~~~

To run the tests, simply run:

.. code-block:: console

    nose2 -F opveclib.examples.tensorflow_clustering opveclib.examples

This should take about 5 minutes on a CPU-only installation or 15 minutes on a GPU-enabled installation.


Documentation
-------------
.. TODO: Provide link to published docs.

