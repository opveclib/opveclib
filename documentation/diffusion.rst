Diffusion
=========

This example implements an anisotropic image diffusion process. We chose it to illustrate the definition of several
common operators used in image processing, such as a 2D Gaussian or 2D filtering. This example also includes an
efficient solver for 2D orthogonal, sparse linear systems of equations called Additive Operator Splitting (AOS).
Further details on this solver can be found in

* Weickert, Haar Romeny, Viergever (1993). Efficient and reliable schemes for nonlinear diffusion filtering. In IEEE Transactions on Image Processing 7(3):398--410.

and especially for a parallel implementation in

* Weickert, Zuiderveld, Haar Romeny, Nissen (1997). Parallel implementations of AOS schemes: a fast way of nonlinear diffusion filtering. In Proc. of Int. Conf. on Image Processing 3:396--399.

Setup
-----
If you want to run the main method for clustering you have to install the following dependency, which will install
matplotlib for python 2.7.

.. code-block:: console

   sudo apt-get install python-matplotlib

If you want to install matplotlib for python 3.4 execute the following:

.. code-block:: console

   pip3 install matplotlib

If not done so, you have to install the development tools to compile matplotlib. This is done through the command:

.. code-block:: console

   sudo apt-get install python3-dev

For correct display of figures under python 3 you need to install the cairocffi package. Before compiling the package
you need to install the dependency on libffi-dev.

.. code-block:: console

   sudo apt-get install libffi-dev

The install cairocffi through the pip install for python 3.

.. code-block:: console

   pip3 install cairocffi



Testing
-------
For testing we download the MRI scan from wikipedia https://commons.wikimedia.org/wiki/File:Head_MRI,_enlarged_inferior_sagittal_sinus.png
and store it in the temporary directory. Next, we load the image and convert into a gray-valued image. Even though the
original image appears gray-valued, however, it is stored in RGB color space. There are two implementations of
the image diffusion method. One implementation uses numpy only and the other uses Operator Vectorization Library (OVL)
operators and tensorflow operators. We compare the computed image diffusion results for equality.

Notice that this example uses the following operators that are defined as separate classes and corresponding numpy
functions:

1. Add boundary of one pixel size to a 2D array.

2. Copy boundary values from one pixel inward to the edges of the 2D array.

3. Delete boundary of one pixel size from a 2D array.

4. Construct a Gaussian 2D filter.

5. Filter 2D data, which is inefficient compared to tensorflow's / CUDA's implementation of filtering.

6. Compute image gradients for the diffusion process.

7. Solve a sparse linear equation system in rows of 2D data.

8. Solve a sparse linear equation system in columns of 2D data.

9. Convert a RGB color image into a gray-valued image.

Except for 9. all these operators have test cases as well using randomly generated data. For our machines with 2xE5-2650
Intel Xeons and GTX Titan Black GPU the test take about four minutes to complete.
