# Copyright 2016 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import math as mt
import numpy as np
import tensorflow as tf
import opveclib as ops
import sys
import warnings


def diffusion2DGPU(image, dt, l, s, nIter):
    """Applies a diffusion process to an image for segmentation on an GPU.

    Assume the image function is f(x), then the image diffusion process is defined by:

        dt u = div( g( |grad u_s|^2 ) grad u

        for u_s = gauss(s) * u  and  grad = (dx,dy)

        with the initial value u(x,t=0) = f(x) and the boundary conditions:

        dn u = 0 for the all x e boundary.

    :param image: The gray-value 2D input image.
    :type image: numpy array.
    :param dt: The step width.
    :type dt: float.
    :param l: Parameter that defines diffusity.
    :type l: float.
    :param s: Parameter sigma of a Gaussian filter kernel.
    :type s: float.
    :param nIter: Number of iterations for the diffusion.
    :type nIter: int.
    :return An image that is the diffused.

    Further details:
    Weickert, Haar Romeny, Viergever (1993). Efficient and reliable schemes for nonlinear diffusion filtering. In IEEE
        Transactions on Image Processing 7(3):398--410.
    Weickert, Zuiderveld, Haar Romeny, Nissen (1997). Parallel implementations of AOS schemes: a fast way of nonlinear
        diffusion filtering. In Proc. of Int. Conf. on Image Processing 3:396--399.
    """
    assert len(image.shape)==2 , "Only gray-value images are supported but found %d channels." % (len(image.shape))
    nGauss  = 2 * int(mt.ceil(s*2)) + 1 # Number of pixels.
    nDim    = 2
    l2      = l*l
    with tf.Session() as sess: # Uses the default GPU
        I = AddBoundaryOp(image).as_tensorflow()
        Gauss = Gauss2DOp(dimOut=[nGauss, nGauss]).as_tensorflow()

        for iter in range(nIter):
            G =  Filter2DOp(I, Gauss).as_tensorflow()

            GRowPlus, GRowMinus, GColPlus, GColMinus = DiffusionGradient2DOp(G, l2=l2).as_tensorflow()

            AlphaRow    = 1 - (- (GRowPlus + GRowMinus)) * nDim * dt
            BetaRow     = - GRowMinus * nDim * dt
            GammaRow    = - GRowPlus * nDim * dt

            AlphaCol    = 1 - (- (GColPlus + GColMinus)) * nDim * dt
            BetaCol     = - GColMinus * nDim * dt
            GammaCol    = - GColPlus * nDim * dt

            I = SolveDiagRow2DOp(AlphaRow, BetaRow, GammaRow, I).as_tensorflow() \
              + SolveDiagCol2DOp(AlphaCol, BetaCol, GammaCol, I).as_tensorflow()
            I = I / nDim
            I = CopyBoundaryOp(I).as_tensorflow()

        I = DelBoundaryOp(I).as_tensorflow()

    return sess.run(I)

def diffusion2DNp(image, dt, l, s, nIter):
    """Applies a diffusion process to an image for segmentation on an CPU using numpy.

    Assume the image function is f(x), then the image diffusion process is defined by:

        dt u = div( g( |grad u_s|^2 ) grad u

        for u_s = gauss(s) * u  and  grad = (dx,dy)

        with the initial value u(x,t=0) = f(x) and the boundary conditions:

        dn u = 0 for the all x e boundary.

    :param image: The gray-value 2D input image.
    :type image: numpy array.
    :param dt: The step width.
    :type dt: float.
    :param l: Parameter that defines diffusity.
    :type l: float.
    :param s: Parameter sigma of a Gaussian filter kernel.
    :type s: float.
    :param nIter: Number of iterations for the diffusion.
    :type nIter: int.
    :return An image that is the diffused.

    Further details:
    Weickert, Haar Romeny, Viergever (1993). Efficient and reliable schemes for nonlinear diffusion filtering. In IEEE
        Transactions on Image Processing 7(3):398--410.
    Weickert, Zuiderveld, Haar Romeny, Nissen (1997). Parallel implementations of AOS schemes: a fast way of nonlinear
        diffusion filtering. In Proc. of Int. Conf. on Image Processing 3:396--399.
    """
    assert len(image.shape)==2, "Only gray-value images are supported but found %d channels." % (len(image.shape))
    nGauss = 2 * int(mt.ceil(s*2)) + 1 # Number of pixels.
    nDim = 2
    l2 = l*l
    I = addBoundaryNp(image)
    Gauss = gauss2DNp(dimOut=[nGauss, nGauss])
    for iter in range(nIter):
        G = filter2DNp(I, Gauss)
        GRowPlus, GRowMinus, GColPlus, GColMinus = diffusionGradient2DNp(G, l2=l2)
        AlphaRow    = 1 - (- (GRowPlus + GRowMinus)) * nDim * dt
        BetaRow     = - GRowMinus * nDim * dt
        GammaRow    = - GRowPlus * nDim * dt

        AlphaCol    = 1 - (- (GColPlus + GColMinus)) * nDim * dt
        BetaCol     = - GColMinus * nDim * dt
        GammaCol    = - GColPlus * nDim * dt

        I = solveDiagRow2DNp(AlphaRow, BetaRow, GammaRow, I) \
          + solveDiagCol2DNp(AlphaCol, BetaCol, GammaCol, I)
        I = I / nDim
        I = copyBoundaryNp(I)

    return delBoundaryNp(I)


class TensorToFloat64(ops.Operator):
    """
    Convert a tensor from any type into a float64 type using the ops.cast function.
    """
    def op(self, dataIn):
        """The definition of the operator.

        :param dataIn: Input tensor.
        :type dataIn: numpy array.
        :return The converted input tensor.
        """
        pos             = ops.position_in(dataIn.shape)
        dataOut         = ops.output(dataIn.shape, ops.float64)
        dataOut[pos]    = ops.cast(dataIn[pos], ops.float64)

        return dataOut

class AddBoundaryOp(ops.Operator):
    """Adds a one pixel boundary to a 2D field.

    This class defines an operator to add a boundary to a 2D field.

    Example:
                 / 3 4 4 \
        dataIn = \ 2 1 0 /


                  / 3 3 4 4 4 \
        dataOut = | 3 3 4 4 4 |
                  | 2 2 1 0 0 |
                  \ 2 2 1 0 0 /
    """
    def op(self, dataIn):
        """The definition of the operator function.

        :param dataIn: 2D data input.
        :type dataIn: numpy array.
        :return 2D data output with added boundary.
        """
        assert len(dataIn.shape)==2 , "Only 2D data is supported but found %d dimensions." \
                                      % (len(dataIn.shape))
        nYIn    = dataIn.shape[0]
        nXIn    = dataIn.shape[1]
        assert nYIn>1, "2D data has %d rows, but must have more than %d rows." % (nYIn, 2)
        assert nXIn>1, "2D data has %d columns, but must have more than %d columns" % (nXIn, 2)
        nYOut   = nYIn + 2
        nXOut   = nXIn + 2
        dataOut = ops.output([nYOut, nXOut], dataIn.dtype)
        pos     = ops.position_in(4)[0] # Work in stripes of 4 threads for a better memory access pattern.
        nY0     = nYIn*pos/4
        nY1     = nYIn*(pos+1)/4

        # Copy the stripe in a row including the boundary values.
        for iY in ops.arange(nY0,nY1):
            dataOut[iY+1, 0] = dataIn[iY, 0]              # left boundary
            for iX in ops.arange(0,nXIn):
                dataOut[iY+1, iX+1] = dataIn[iY, iX]
            dataOut[iY+1, nXOut-1] = dataIn[iY, nXIn-1]   # right boundary

        # The first thread works on the top boundary.
        with ops.if_(pos==0):
            dataOut[0, 0] = dataIn[0, 0]            # top,left corner
            for iX in ops.arange(0, nXIn):
                dataOut[0, iX+1] = dataIn[0, iX]      # top boundary
            dataOut[0, nXOut-1] = dataIn[0, nXIn-1] # top, right corner

        # The last thread works on the bottom boundary.
        with ops.elif_(pos==3):
            dataOut[nYOut-1, 0] = dataIn[nYIn-1, 0]             # bottom left corner
            for iX in ops.arange(0, nXIn):
                dataOut[nYOut-1, iX+1] = dataIn[nYIn-1, iX]       # bottom boundary
            dataOut[nYOut-1, nXOut-1] = dataIn[nYIn-1, nXIn-1]  # bottom right corner

        return dataOut


def addBoundaryNp(dataIn):
    """Adds a one pixel boundary to the input field.
       This is a reference implementation of the add boundary operator in python.

       :param dataIn: 2D data input.
       :type dataIn: numpy array.
       :return 2D data output with added boundary.

    """
    nYIn    = dataIn.shape[0]
    nXIn    = dataIn.shape[1]
    nYOut   = nYIn + 2
    nXOut   = nXIn + 2
    dataOut = np.zeros((nYOut, nXOut))

    for iY in range(0,nYIn):
        dataOut[iY+1, 0] = dataIn[iY, 0]              # left boundary
        for iX in range(0,nXIn):
            dataOut[iY+1, iX+1] = dataIn[iY, iX]
        dataOut[iY+1, nXOut-1] = dataIn[iY, nXIn-1]   # right boundary

    for iX in range(0, nXIn):
        dataOut[0, iX+1] = dataIn[0, iX]              # top boundary
        dataOut[nYOut-1, iX+1] = dataIn[nYIn-1, iX]   # bottom boundary

    # Four corners
    dataOut[0, 0]                = dataIn[0, 0]
    dataOut[0, nXOut-1]          = dataIn[0, nXIn-1]
    dataOut[nYOut-1, 0]          = dataIn[nYIn-1, 0]
    dataOut[nYOut-1, nXOut-1]    = dataIn[nYIn-1, nXIn-1]

    return dataOut


class DelBoundaryOp(ops.Operator):
    """Deletes values from a one pixel wide boundary for a 2D field.

    This class defines an operator to delete the boundary values of a 2D field.

    Example:
                 / 1 0 2 3 5 \
        dataIn = | 9 3 4 4 2 |
                 | 1 2 1 0 7 |
                 \ 5 3 2 2 8 /

                  / 3 4 4 \
        dataOut = \ 2 1 0 /
    """
    def op(self, dataIn):
        """The definition of the operator function.

        :param dataIn: 2D data input.
        :return 2D data output with deleted boundary.
        """
        assert len(dataIn.shape)==2 , "Only 2D data is supported but found %d dimensions." % (len(dataIn.shape))

        nYIn    = dataIn.shape[0]
        nXIn    = dataIn.shape[1]

        assert nYIn>2, "Data must have at least %d rows but has %d rows." %(3, nYIn)
        assert nXIn>2, "Data must have at least %d columns but has %d columns." %(3, nXIn)

        nYOut   = nYIn-2
        nXOut   = nXIn-2
        dataOut = ops.output([nYOut, nXOut], dataIn.dtype)
        pos     = ops.position_in(4)[0] # Work in stripes for a better memory access pattern.
        nY0     = nYOut*pos/4
        nY1     = nYOut*(pos+1)/4
        # Copy the stripe in a row, excluding boundary values.
        for iY in ops.arange(nY0,nY1):
            for iX in ops.arange(0,nXOut):
                dataOut[iY, iX] = dataIn[1+iY, 1+iX]

        return dataOut

def delBoundaryNp(dataIn):
    """Deletes a one pixel boundary in the input field.

    This is a reference implementation of the operator function in python.

    :param dataIn: 2D data input.
    :type dataIn: numpy array.
    :return 2D data output with deleted boundary.
    """
    nYIn    = dataIn.shape[0]
    nXIn    = dataIn.shape[1]
    nYOut   = nYIn - 2
    nXOut   = nXIn - 2
    dataOut = np.zeros((nYOut, nXOut))
    for iY in range(0,nYOut):
        for iX in range(0,nXOut):
            dataOut[iY, iX] = dataIn[1+iY, 1+iX]

    return dataOut


class CopyBoundaryOp(ops.Operator):
    """Copies values from inside the boundary to the one pixel boundary for 2D field.

    This class defines an operator to copy the boundary values of a 2D field.

    Example:
                 / 1 0 2 3 5 \
        dataIn = | 9 3 4 4 2 |
                 | 1 2 1 0 7 |
                 \ 5 3 2 2 8 /

                  / 3 3 4 4 4 \
        dataOut = | 3 3 4 4 4 |
                  | 2 2 1 0 0 |
                  \ 2 2 1 0 0 /
    """
    def op(self, dataIn):
        """The definition of the operator function.

        :param dataIn: 2D data input.
        :type dataIn: numpy array.
        :return 2D data output with copied boundary.
        """
        assert len(dataIn.shape)==2 , "Only 2D data is supported but found %d dimensions." % (len(dataIn.shape))

        nY  = dataIn.shape[0]
        nX  = dataIn.shape[1]

        assert nY>1, "2D data has %d rows, but must have more than %d rows." % (nY, 2)
        assert nX>1, "2D data has %d columns, but must have more than %d columns" % (nX, 2)

        dataOut = ops.output(dataIn.shape, dataIn.dtype)
        pos     = ops.position_in(4)[0] # Work in stripes for a better memory access pattern.
        nY0     = 1+(nY-2)*pos/4
        nY1     = 1+(nY-1)*(pos+1)/4

        # Copy the stripe in a row including the boundary values.
        for iY in ops.arange(nY0,nY1):
            dataOut[iY, 0] = dataIn[iY, 1]          # left boundary
            for iX in ops.arange(1,nX-1):
                dataOut[iY, iX] = dataIn[iY, iX]
            dataOut[iY, nX-1] = dataIn[iY, nX-2]    # right boundary

        # The first thread will work on the top boundary.
        with ops.if_(pos==0):
            dataOut[0, 0] = dataIn[1, 1]        # top,left corner
            for iX in ops.arange(1, nX-1):
                dataOut[0, iX] = dataIn[1, iX]  # top boundary
            dataOut[0, nX-1] = dataIn[1, nX-2]  # top, right corner

        # The last thread will work on the bottom boundary.
        with ops.elif_(pos==3):
            dataOut[nY-1, 0] = dataIn[nY-2, 1]          # bottom left corner
            for iX in ops.arange(1, nX-1):
                dataOut[nY-1, iX] = dataIn[nY-2, iX]    # bottom boundary
            dataOut[nY-1, nX-1] = dataIn[nY-2, nX-2]    # bottom right corner

        return dataOut


def copyBoundaryNp(dataIn):
    """Copies a one pixel boundary in the input field.

    This is a reference implementation of the operator function in python.

    :param dataIn: 2D data input.
    :type dataIn: numpy array.
    :return 2D data output with copied boundary.
    """
    nY = dataIn.shape[0]
    nX = dataIn.shape[1]
    dataOut = np.zeros((nY, nX))
    for iY in range(1,nY-1):
        dataOut[iY, 0] = dataIn[iY, 1]          # left boundary
        for iX in range(1,nX-1):
            dataOut[iY, iX] = dataIn[iY, iX]
        dataOut[iY, nX-1] = dataIn[iY, nX-2]    # right boundary

    for iX in range(1, nX-1):
        dataOut[0, iX] = dataIn[1, iX]          # top boundary
        dataOut[nY-1, iX] = dataIn[nY-2, iX]    # bottom boundary

    # Four corner values.
    dataOut[0, 0]       = dataIn[1, 1]
    dataOut[0, nX-1]    = dataIn[1, nX-2]
    dataOut[nY-1, 0]    = dataIn[nY-2, 1]
    dataOut[nY-1, nX-1] = dataIn[nY-2, nX-2]

    return dataOut


class Gauss2DOp(ops.Operator):
    """Gaussian 2D kernel.

    Define a 2D Gaussian kernel.

    g(x,y) = 1/(2 pi sigmaY sigmaX) exp(-1/2*((x-muX)/sigmaX)^2 -1/2((y-muY)/sigmaY)^2)

    In the implementation we normalize each entry in the filter kernel by the sum of all entries, rather than using the
    analytical form of 1/(2 pi sigmaY sigmaX).
    """
    def op(self, dimOut):
        """The definition of the operator function.

        :param dimOut: Output dimensions [nY, nX].
        :type dimOut: list.
        :return 2D Gaussian kernel with numeric normalization.
        """
        dataOut = ops.output(dimOut, ops.float64)
        n0      = dimOut[0]
        n1      = dimOut[1]
        ops.position_in(1) # Workgroup size of 1.
        accum   = ops.variable(0, ops.float64)
        nHalf0  = ops.variable(n0/2.0, ops.float64)
        sigma0  = ops.variable(n0/4.0, ops.float64)
        nHalf1  = ops.variable(n1/2.0, ops.float64)
        sigma1  = ops.variable(n1/4.0, ops.float64)
        data    = ops.zeros(dimOut, ops.float64)

        # A single thread does all the work.
        for iElem0 in ops.arange(n0):
            for iElem1 in ops.arange(n1):
                x = ops.cast(iElem0, ops.float64) - nHalf0
                y = ops.cast(iElem1, ops.float64) - nHalf1
                value = ops.exp(- x*x/(2*sigma0*sigma0) - y*y/(2*sigma1*sigma1))
                accum <<= accum + value
                data[iElem0,iElem1] = value

        # Normalize the 2D Gaussian.
        for iElem0 in ops.arange(n0):
            for iElem1 in ops.arange(n1):
                dataOut[iElem0,iElem1] = data[iElem0,iElem1]/accum

        return dataOut


def gauss2DNp(dimOut):
    """Gaussian 2D kernel.

    This is a reference implementation of the operator function in python.

    :param dimOut: Output dimensions [nY, nX].
    :type dimOut: list.
    :return 2D Gaussian kernel.
    """
    n0      = dimOut[0]
    n1      = dimOut[1]
    nHalf0  = n0/2.0
    sigma0  = n0/4.0
    nHalf1  = n1/2.0
    sigma1  = n1/4.0
    accum   = 0.0
    dataOut = np.zeros((n0, n1))

    # Compute values in 2D kernel for Gaussian.
    for i0 in range(0,n0):
        for i1 in range(0,n1):
            value = mt.exp(- (i0-nHalf0)*(i0-nHalf0)/(2*sigma0*sigma0)
                           - (i1-nHalf1)*(i1-nHalf1)/(2*sigma1*sigma1))
            accum = accum + value
            dataOut[i0,i1] = value

    # Normalize the 2D Gaussian.
    for i0 in range(0,n0):
        for i1 in range(0,n1):
            dataOut[i0,i1] = dataOut[i0,i1]/accum

    return dataOut

class Filter2DOp(ops.Operator):
    """Filtering for 2D input data and kernels using the circular boundary condition.

    The filtering function is not an efficient implementation for large kernels, typically larger than 7 x 7 pixels.
    """
    def op(self, dataIn, kernelIn):
        """The definition of the operator function.

        :param dataIn: 2D data input.
        :type dataIn: numpy array.
        :param kernelIn: 2D filtering kernel.
        :type kernelIn: numpy array.
        :return Filtered 2D data.
        """
        assert(len(dataIn.shape) == 2)
        assert(dataIn.shape[0] >= kernelIn.shape[0]) # data input must be larger than kernel
        assert(dataIn.shape[1] >= kernelIn.shape[1])

        dataOut = ops.output(dataIn.shape, dataIn.dtype)

        nYIn = dataIn.shape[0]
        nXIn = dataIn.shape[1]
        nYKl = kernelIn.shape[0]
        nXKl = kernelIn.shape[1]

        pos = ops.position_in(dataIn.shape) # Each position within the input image.

        iY = ops.cast(pos[0] - (nYKl-1)/2 + nYKl*nYIn, ops.int64) # Add offset for proper wrapping if nYH > nYIn.
        iX = ops.cast(pos[1] - (nXKl-1)/2 + nXKl*nXIn, ops.int64)

        accum = ops.variable(0, ops.float64)

        # Sum all values for this position of the filter kernel in the input image.
        for iYKl in ops.arange(nYKl):
            for iXKl in ops.arange(nXKl):
                # Casted iY and iX as int64 above since iYKl and iXKl are dynamic int64s in this loop.
                iiY = (iY + iYKl) % nYIn
                iiX = (iX + iXKl) % nXIn

                # Assign to a scalar with the <<= operator.
                accum <<= accum + dataIn[iiY, iiX]*kernelIn[iYKl, iXKl]

        # Assign to an output field [] with the '=' operator.
        dataOut[pos] = accum

        return dataOut

def filter2DNp(dataIn, kernelIn):
    """Filtering for 2D input data and kernels using the circular boundary condition.

    This is a reference implementation of the operator function in python.

    :param dataIn: 2D data input.
    :type dataIn: numpy array.
    :param kernelIn: 2D filtering kernel.
    :type kernelIn: numpy array.
    :return Filtered 2D data.
    """
    nYIn    = dataIn.shape[0]
    nXIn    = dataIn.shape[1]
    nYKl    = kernelIn.shape[0]
    nXKl    = kernelIn.shape[1]
    nYH     = int((nYKl-1)/2 - nYKl*nYIn) # Add offset for proper wrapping if nYH > nYIn.
    nXH     = int((nXKl-1)/2 - nXKl*nXIn)
    dataOut = np.zeros((nYIn,nXIn))
    for iYIn in range(0,nYIn):
        for iXIn in range(0,nXIn):
            accum = 0.0
            for iYKl in range(0,nYKl):
                for iXKl in range(0,nXKl):
                    iY = (iYIn + iYKl - nYH) % nYIn
                    iX = (iXIn + iXKl - nXH) % nXIn
                    accum += dataIn[iY,iX]*kernelIn[iYKl,iXKl]

            dataOut[iYIn,iXIn] = accum

    return dataOut



class DiffusionGradient2DOp(ops.Operator):
    """Computes the gradient for image diffusion.

    The method uses a forward (plus) and backward (minus) difference to compute the gradient of the provided image.
    """
    def op(self, image, l2):
        """The definition of the operator function.

        :param image: An 2D gray-value input image.
        :type image: numpy array.
        :param l2: The lambda x lambda parameter.
        :type l2: float.
        """
        assert len(image.shape)==2 , "Only 2D data is supported but found %d dimensions." % (len(image.shape))
        nY = image.shape[0]
        nX = image.shape[1]

        gradRowPlus     = ops.output_like(image)
        gradRowMinus    = ops.output_like(image)
        gradColPlus     = ops.output_like(image)
        gradColMinus    = ops.output_like(image)

        pos     = ops.position_in(image.shape)
        iY      = pos[0]
        iX      = pos[1]

        dyPlus  = ops.variable(image[(iY+1)%nY, iX] - image[iY, iX], image.dtype)
        dxPlus  = ops.variable(image[iY, (iX+1)%nX] - image[iY, iX], image.dtype)
        dyMinus = ops.variable(image[iY,iX] - image[(iY-1+nY)%nY, iX], image.dtype)
        dxMinus = ops.variable(image[iY,iX] - image[iY, (iX-1+nX)%nX], image.dtype)

        gradRowPlus[iY,iX]  = 1.0/(1.0 + dyPlus*dyPlus/l2)
        gradRowMinus[iY,iX] = 1.0/(1.0 + dyMinus*dyMinus/l2)
        gradColPlus[iY,iX]  = 1.0/(1.0 + dxPlus*dxPlus/l2)
        gradColMinus[iY,iX] = 1.0/(1.0 + dxMinus*dxMinus/l2)

        return gradRowPlus, gradRowMinus, gradColPlus, gradColMinus

def diffusionGradient2DNp(image, l2):
    """Computes the gradient for image diffusion.

    :param image: An 2D gray-value input image.
    :type image: numpy array.
    :param l2: The lambda x lambda parameter.
    :type l2: float.
    """
    assert len(image.shape)==2 , "Only 2D data is supported but found %d dimensions." % (len(image.shape))

    nY = image.shape[0]
    nX = image.shape[1]

    gradRowPlus     = np.zeros((nY, nX))
    gradRowMinus    = np.zeros((nY, nX))
    gradColPlus     = np.zeros((nY, nX))
    gradColMinus    = np.zeros((nY, nX))

    for iY in range(0, nY):
        for iX in range(0, nX):
            dyPlus  = image[(iY+1)%nY, iX] - image[iY, iX]
            dxPlus  = image[iY, (iX+1)%nX] - image[iY, iX]
            dyMinus = image[iY,iX] - image[(iY-1+nY)%nY, iX]
            dxMinus = image[iY,iX] - image[iY, (iX-1+nX)%nX]

            gradRowPlus[iY,iX]  = 1.0/(1.0 + dyPlus*dyPlus/l2)
            gradRowMinus[iY,iX] = 1.0/(1.0 + dyMinus*dyMinus/l2)
            gradColPlus[iY,iX]  = 1.0/(1.0 + dxPlus*dxPlus/l2)
            gradColMinus[iY,iX] = 1.0/(1.0 + dxMinus*dxMinus/l2)

    return gradRowPlus, gradRowMinus, gradColPlus, gradColMinus



class SolveDiagRow2DOp(ops.Operator):
    """Solves a sparse linear equation system of the form: gamma(i-1) x(i-1) + alpha(i) x(i) + beta(i+1) x(i+1) = Y(i).
    The solution is computed for ROWS.

    In this case the solution for the rows is computed while parallelizing the computation over columns.
    """
    def op(self, alpha, beta, gamma, y):
        """Solving the linear equation systems for rows in the 2D matrix.

        :param alpha: Coefficient of the sparse linear equations system.
        :type alpha: numpy array.
        :param beta: Coefficient.
        :type beta: numpy array.
        :param gamma: Coefficient.
        :type gamma: numpy array.
        :param y: Input values.
        :type y: numpy array.
        :return Solution x of gamma(i-1) x(i-1) + alpha(i) x(i) + beta(i+1) x(i+1) = Y(i).
        """
        assert len(y.shape)==2 , "Only 2D data is supported but found %d dimensions." % (len(y.shape))
        nY  = y.shape[0]
        nX  = y.shape[1]
        m   = ops.zeros(nY, y.dtype)
        l   = ops.zeros(nY, y.dtype)
        w   = ops.zeros(nY, y.dtype)
        x   = ops.zeros(nY, y.dtype)
        z   = ops.output([nY, nX], y.dtype)
        iX  = ops.position_in(nX)[0]

        # Initialize the 0th row.
        m[0] = alpha[0,iX]
        for iY in ops.arange(0,nY-1):
            l[iY]    = gamma[iY,iX]/(m[iY]+sys.float_info.epsilon)
            m[iY+1]  = alpha[iY+1,iX] - l[iY]*beta[iY+1,iX]

        # Forward substitution (L W = Y)
        w[0] = y[0,iX]
        for iY in ops.arange(1,nY):
            w[iY] = y[iY,iX] - l[iY-1]*w[iY-1]

        # Backward substitution (R X = W)
        x[nY-1] = w[nY-1]/m[nY-1]
        for iY in ops.arange(nY-2, -1, -1):
            x[iY] = (w[iY] - beta[iY+1,iX]*x[iY+1])/(m[iY]+sys.float_info.epsilon)

        # Copy to the output z.
        for iY in ops.arange(0,nY):
            z[iY,iX] = x[iY]

        return z

def solveDiagRow2DNp(alpha, beta, gamma, y):
    """Solves a sparse linear equation system of the form: gamma(i-1) x(i-1) + alpha(i) x(i) + beta(i+1) x(i+1) = Y(i).
    The solution is computed for ROWS.

    This is a reference implementation for the solve diag operator function in python. For this implementation there is
    no parallelization.

    :param alpha: Coefficient of the sparse linear equations system.
    :type aplha: numpy array.
    :param beta: Coefficient.
    :type beta: numpy array.
    :param gamma: Coefficient.
    :type gamma: numpy array.
    :param y: Input values.
    :type y: numpy array.
    :return Solution x of gamma(i-1) x(i-1) + alpha(i) x(i) + beta(i+1) x(i+1) = Y(i).
    """
    assert len(y.shape)==2 , "Only 2D data is supported but found %d dimensions." % (len(y.shape))
    nY  = y.shape[0]
    nX  = y.shape[1]
    m   = np.zeros([nY, nX], y.dtype)
    l   = np.zeros([nY, nX], y.dtype)
    w   = np.zeros([nY, nX], y.dtype)
    x   = np.zeros([nY, nX], y.dtype)

    # Initialize 0th row in m.
    for iX in range(0,nX):
        m[0,iX] = alpha[0,iX]

    for iY in range(0,nY-1):
        for iX in range(0,nX):
            l[iY,iX]    = gamma[iY,iX]/(m[iY,iX]+sys.float_info.epsilon)
            m[iY+1,iX]  = alpha[iY+1,iX] - l[iY,iX]*beta[iY+1,iX]

    for iX in range(0,nX):
        w[0,iX] = y[0,iX]

    # Forward substitution (L W = Y)
    for iY in range(1,nY):
        for iX in range(0,nX):
            w[iY,iX] = y[iY,iX] - l[iY-1,iX]*w[iY-1,iX]

    # Backward substitution (R X = W)
    for iX in range(0,nX):
        x[nY-1,iX] = w[nY-1,iX]/m[nY-1,iX]

    # Copy to the output z.
    for iY in range(nY-2, -1, -1):
        for iX in range(0,nX):
            x[iY,iX] = (w[iY,iX] - beta[iY+1,iX]*x[iY+1,iX])/(m[iY,iX] + sys.float_info.epsilon)

    return x


class SolveDiagCol2DOp(ops.Operator):
    """Solves a sparse linear equation system of the form: gamma(i-1) x(i-1) + alpha(i) x(i) + beta(i+1) x(i+1) = Y(i).
    The solution is computed for COLUMNS.

    In this case the solution for the columns is computed while parallelizing the computation over rows.
    """
    def op(self, alpha, beta, gamma, y):
        """Solving the linear equation systems for columns in the 2D matrix.

        :param alpha: Coefficient of the sparse linear equations system.
        :type alpha: numpy array.
        :param beta: Coefficient.
        :type beta: numpy array.
        :param gamma: Coefficient.
        :type gamma: numpy array.
        :param y: Input values.
        :type y: numpy array.
        :return Solution x of gamma(i-1) x(i-1) + alpha(i) x(i) + beta(i+1) x(i+1) = Y(i).
        """
        assert len(y.shape)==2 , "Only 2D data is supported but found %d dimensions." % (len(y.shape))
        nY  = y.shape[0]
        nX  = y.shape[1]
        m   = ops.zeros(nX, y.dtype)
        l   = ops.zeros(nX, y.dtype)
        w   = ops.zeros(nX, y.dtype)
        x   = ops.zeros(nX, y.dtype)
        z   = ops.output([nY, nX], y.dtype)
        iY  = ops.position_in(nY)[0]

        m[0] = alpha[iY,0]
        for iX in ops.arange(0,nX-1):
            l[iX]    = gamma[iY,iX]/(m[iX]+sys.float_info.epsilon)
            m[iX+1]  = alpha[iY,iX+1] - l[iX]*beta[iY,iX+1]

        # Forward substitution (L W = Y)
        w[0] = y[iY,0]
        for iX in ops.arange(1,nX):
            w[iX] = y[iY,iX] - l[iX-1]*w[iX-1]

        # Backward substitution (R X = W)
        x[nX-1] = w[nX-1]/m[nX-1]
        for iX in ops.arange(nX-2, -1, -1):
            x[iX] = (w[iX] - beta[iY,iX+1]*x[iX+1])/(m[iX]+sys.float_info.epsilon)

        # Copy to the output z.
        for iX in ops.arange(0,nX):
            z[iY,iX] = x[iX]

        return z


def solveDiagCol2DNp(alpha, beta, gamma, y):
    """Solves a sparse linear equation system of the form: gamma(i-1) x(i-1) + alpha(i) x(i) + beta(i+1) x(i+1) = Y(i).

    This is a reference implementation for the operator function in python. For this implementation there is no
    parallelization.

    :param alpha: Coefficient of the sparse linear equations system.
    :type alpha: numpy array.
    :param beta: Coefficient.
    :type beta: numpy array.
    :param gamma: Coefficient.
    :type gamma: numpy array.
    :param y: Input values.
    :type y: numpy array.
    :return Solution x of gamma(i-1) x(i-1) + alpha(i) x(i) + beta(i+1) x(i+1) = Y(i).
    """
    assert len(y.shape)==2 , "Only 2D data is supported but found %d dimensions." % (len(y.shape))
    nY  = y.shape[0]
    nX  = y.shape[1]
    m   = np.zeros([nY, nX], y.dtype)
    l   = np.zeros([nY, nX], y.dtype)
    w   = np.zeros([nY, nX], y.dtype)
    x   = np.zeros([nY, nX], y.dtype)

    for iY in range(0,nY):
        m[iY,0] = alpha[iY,0]
        for iX in range(0,nX-1):
            l[iY,iX]    = gamma[iY,iX]/(m[iY,iX]+sys.float_info.epsilon)
            m[iY,iX+1]  = alpha[iY,iX+1] - l[iY,iX]*beta[iY,iX+1]

        # Forward substitution (L W = Y)
        w[iY,0] = y[iY,0]
        for iX in range(1,nX):
            w[iY,iX] = y[iY,iX] - l[iY,iX-1]*w[iY,iX-1]

        # Backward substitution (R X = W)
        x[iY,nX-1] = w[iY,nX-1]/m[iY,nX-1]
        for iX in range(nX-2, -1, -1):
            x[iY,iX] = (w[iY,iX] - beta[iY,iX+1]*x[iY,iX+1])/(m[iY,iX]+sys.float_info.epsilon)

    return x
