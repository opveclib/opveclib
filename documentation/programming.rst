Programmer's Guide
==================


Introduction
------------

The mission of OVL is to provide a framework for easily defining high-performance, python-native
operators that can work transparently with TensorFlow on a variety of different processor architectures. OVL operators
are written by users in OVL's python-embedded domain specific language (DSL). The programming model of operators
is similar to that of CUDA or OpenCL - conceptually an operator is a stateless function that is mapped over
a set of user-defined thread or worker indices. Operators are defined as the python interpreter encounters them but
are evaluated lazily, allowing for numerical and performance optimizations to be applied to the entire graph of
defined operators before evaluation time.

The OVL DSL is designed to strike a balance between productivity, performance, and portability and as such there
are some important restrictions that differentiate it from similar approaches:

* The DSL is strongly typed and tensor shapes are part of the type system. This means that inputs and outputs must
  have a concrete shape - there is no support for dynamic tensor sizing.
* Individual worker elements cannot communicate and, as such, there are no synchronization barriers.
* Operators have no persistent state between calls.

Troubleshooting
~~~~~~~~~~~~~~~
General usage and setup questions should be posted to
`StackOverflow <https://stackoverflow.com/questions/tagged/opveclib>`__. Please post bugs and feature requests to
`GitHub <https://github.com/opveclib/opveclib/issues>`__ with as much information as possible including:

* System environment information
* Library versions ``pip show tensorflow opveclib``
* Logs, e.g. by using ``logging.basicConfig(filename='output.log',level=logging.DEBUG)`` in your code
* Instructions or code snippets for reproducing the issue

Common Pitfalls
~~~~~~~~~~~~~~~

OVL currently uses the python interpreter to parse operators into its own intermediate representation, an approach
which comes with some limitations. These limitations result in a programming API that has two common pitfalls that
raise a syntax error when encountered. These pitfalls are highlighted here to prevent confusion, but examples of
proper use cases are detailed in the examples section below. These limitations may eventually disappear (i.e. not
throw a syntax error and work as expected) as improvements are made to library internals.

Assignment
__________

Once variables are defined within the operator, the python-native assignment operator `` = `` cannot be used on them.
Use the `` <<= `` operator instead. For example, the following should raise an error:

.. code-block:: python

    import opveclib as ovl
    x = ovl.variable(ovl.float32)
    x = 2.0

The correct way to assign a value to a previously-defined variable is as follows:

.. code-block:: python

    import opveclib as ovl
    x = ovl.variable(ovl.float32)
    x <<= 2.0

Conditionals
____________

The python-native ``if``, ``elif``, and ``else`` statements cannot be used with OVL expressions. OVL
conditionals must be invoked from within a ``with`` statement. More specifically, the following should raise an error:

.. code-block:: python

    import opveclib as ovl
    x = ovl.variable(ovl.float32)
    if x < -1:
        x <<= -1
    elif x > 1:
        x <<= 1

The correct way to use conditionals is as follows:

.. code-block:: python

    import opveclib as ovl
    x = ovl.variable(ovl.float32)
    with ovl.if_(x < -1):
        x <<= -1
    with ovl.elif_(x > 1):
        x <<= 1

Guide by examples
-----------------

Hello World
~~~~~~~~~~~
Designing the simplest OVL operator requires understanding a few key concepts and their corresponding
implementations in the OVL API. This example takes the absolute value of an input tensor and shows all of
the basic functionality necessary to create and use an operator. New concepts introduced here include:

* An OVL operator is defined by creating a python function and decorating it with the ``operator()`` decorator.
* Arguments to the operator are the tensors that it will operate on at evaluation time.
* Output tensors are the only thing that can be returned from operators. They are defined with the ``output`` and
  ``output_like`` functions.
* Operators are implicitly mapped over a set of workgroup positions. The workgroup shape must be statically
  defined based on the arguments to the operator and must be either a single int or a list of ints. The
  ``position_in`` function is used to define the workgroup shape and returns a ``PositionTensor`` object which is
  used to identify the position of the current worker element. A workgroup shape can be defined to be any
  number of dimensions that makes sense for the problem at hand. A PositionTensor can be indexed to obtain the
  current worker's position along a specific axis.
* Data elements in input tensors are accessed by indexing into them. Indexing into an input tensor yields a
  ``Scalar``. A ``Scalar`` can be transformed into a new ``Scalar`` by applying scalar operators to them,
  in this case the ``absolute`` function.
* Output tensors are written to by setting ``Scalar`` values at a specific position, in this case the position is
  just the ``PositionTensor``, but more complicated write patterns follow in other examples.
* Operators can be tested independently of the TensorFlow runtime using the OVL test infrastructure via the
  ``evaluate`` function. Testing operators using this infrastructure is recommended since it isolates the operator
  from the TF runtime. An explicit ``evaluate`` function is used so that operators can be lazily evaluated,
  increasing the opportunity for optimization.
* Operators are linked into the TensorFlow runtime by explicitly converting operator outputs to TensorFlow tensors
  with the ``as_tensorflow`` function.

.. testcode::

    import numpy as np
    import tensorflow as tf
    import opveclib as ovl


    @ovl.operator()
    def absolute(input_tensor):
        # define the output tensor
        output_tensor = ovl.output_like(input_tensor)

        # define the workgroup shape and get workgroup position reference
        wg_position = ovl.position_in(input_tensor.shape)

        # read input element at current workgroup position
        input_element = input_tensor[wg_position]

        # apply absolute function
        abs = ovl.absolute(input_element)

        # set the output element
        output_tensor[wg_position] = abs

        return output_tensor

    # define a numpy input
    in_np = np.arange(-3, 4, dtype=np.float32)
    # apply the operator
    out = absolute(in_np)
    # lazily evaluate with the OVL test infrastructure
    print(ovl.evaluate([out]))

    # explicitly convert to tensorflow tensor
    out_tf = ovl.as_tensorflow(out)
    # lazily evaluate using tensorflow
    sess = tf.Session()
    print(sess.run([out_tf])[0])

Outputs:

.. testoutput::

   [ 3.  2.  1.  0.  1.  2.  3.]
   [ 3.  2.  1.  0.  1.  2.  3.]


Constants
~~~~~~~~~

Many operators depend on an input value that does not change throughout the lifetime of the operator, and is not
dependent on the data values of the input tensors.
Examples include summing along a constant axis, applying a constant threshold,
and applying a constant power to an input tensor. This example implements the ``power`` function which raises an
input tensor to a specified power and introduces the new concept of a constant:

* Constant arguments are differentiated from input tensors by explicitly giving the operator function argument a default
  value. If the default value is ``None`` an error will be raised if the argument is not set when the operator
  is applied. When calling the operator, constants are specified by passing them to the operator as
  keyword arguments.

.. testcode::

    import numpy as np
    import tensorflow as tf
    import opveclib as ovl


    @ovl.operator()
    def power(input_tensor, exponent=None):
        # define the output tensor
        output_tensor = ovl.output_like(input_tensor)

        # define the workgroup shape and get workgroup position reference
        wg_position = ovl.position_in(input_tensor.shape)

        # read input element at current workgroup position
        input_element = input_tensor[wg_position]

        # apply power function and set the output element
        output_tensor[wg_position] = ovl.power(input_element, exponent)

        return output_tensor

    # define a numpy input
    in_np = np.arange(-3, 4, dtype=np.float32)
    # apply the operator
    # Note that constants must be explicitly set as keyword arguments
    out = power(in_np, exponent=2)
    # lazily evaluate with the OVL test infrastructure
    print(ovl.evaluate([out]))

    # explicitly convert to tensorflow tensor
    out_tf = ovl.as_tensorflow(out)
    # lazily evaluate using tensorflow
    sess = tf.Session()
    print(sess.run([out_tf])[0])


Outputs:

.. testoutput::

   [ 9.  4.  1.  0.  1.  4.  9.]
   [ 9.  4.  1.  0.  1.  4.  9.]



Conditionals and Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~

Operators can exhibit control flow and use thread-local memory by using OVL conditionals and variables. This example
implements the ``clip`` function which clips the values of the input tensor to within the specified boundaries.
The following new concepts are illustrated here:

* The ``Variable``, which is a worker-local scalar that has state which can be set with the ``<<=`` operator.
* The conditionals ``with if_()``, ``with elif_()`` and ``with else_()`` which are used to conditionally
  execute a segment of the operator. Python native conditionals can be used from within an operator, but only
  when they are operating on a constant. This example differentiates between the two types of conditional.
* The ``forbid_none_valued_constants`` option to the ``operator`` decorator which overrides the default behavior
  to and allows ``None`` as a valid constant value.

.. testcode::

    import numpy as np
    import opveclib as ovl


    # override default behavior and allow None valued constants
    @ovl.operator(forbid_none_valued_constants=False)
    def clip(arg, threshold1=None, threshold2=None):
        pos = ovl.position_in(arg.shape)
        clipped = ovl.output_like(arg)

        # define a worker local variable
        clipped_val = ovl.variable(0, arg.dtype)

        # assign the value of variables with the the <<= operator
        clipped_val <<= arg[pos]

        # python if statements can be used on constants to control how the operator behaves
        if threshold1 is not None:
            # when using conditionals on OVL expressions, you must use the OVL "with if_" function
            #  these conditionals are evaluated at run time and are input-data dependent
            with ovl.if_(clipped_val < threshold1):
                clipped_val <<= threshold1

        # this python native if statement is used to control how the operator behaves at definition time
        if threshold2 is not None:
            # this conditional controls how the operator deals with run time data
            with ovl.if_(clipped_val > threshold2):
                clipped_val <<= threshold2

        clipped[pos] = clipped_val

        return clipped

    # define a numpy input
    in_np = np.arange(-3, 4, dtype=np.float32)
    # apply the operator
    #  not all constants have to be supplied here since None value constants are permitted
    out1 = clip(in_np, threshold1=-1)
    out2 = clip(in_np, threshold2=1)
    out3 = clip(in_np, threshold1=-1, threshold2=1)
    # lazily evaluate with the OVL test infrastructure
    res1, res2, res3 = ovl.evaluate([out1, out2, out3])
    print(res1)
    print(res2)
    print(res3)

Outputs:

.. testoutput::

    [-1. -1. -1.  0.  1.  2.  3.]
    [-3. -2. -1.  0.  1.  1.  1.]
    [-1. -1. -1.  0.  1.  1.  1.]



Loops and non-local IO
~~~~~~~~~~~~~~~~~~~~~~

OVL supports iterating over ranges and defining arbitrary workgroup shapes that may or may not be the same size
as the input or output tensors. This simple example of a naive implementation of matrix-vector multiplication
show the use of two new concenpts:

* The `arange()` iterator which works like the python native `range` but allows for runtime iteration based off
  of either constants or data from one of the input tensors
* Arbritrary workgroup shapes - in this case the workgroup shape is determined by the number of rows in the matrix.
* Non-local IO is possible, in the sense that in this example each worker reads in multiple elements from the
  input matrix and vector.

.. testcode::

    import numpy as np
    import opveclib as ovl


    @ovl.operator()
    def matmul(mat, vec):
        # assert type properties of the two input tensors
        assert mat.rank == 2
        assert vec.rank == 1
        assert mat.shape[1] == vec.shape[0]
        assert mat.dtype == vec.dtype

        rows = mat.shape[0]
        cols = mat.shape[1]

        # define the output to be one dimensional
        y = ovl.output(rows, mat.dtype)

        # define a number of workers equal to the number of rows
        row = ovl.position_in(rows)[0]

        # define the accumulator and iterate over the matrix columns
        accum = ovl.variable(0, mat.dtype)
        for col in ovl.arange(cols):
            # accumulated the product of each element
            accum <<= accum + mat[row, col]*vec[col]

        y[row] = accum
        return y

    m = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.float32)
    x = np.array([1, 2, 3], dtype=np.float32)
    out = matmul(m, x)
    print(ovl.evaluate([out]))


Outputs:

.. testoutput::

    [  8.  26.  44.]


Worker-local tensors
~~~~~~~~~~~~~~~~~~~~

Similar to the OVL Variable, OVL also supports dynamically declared worker-local tensors of any dimension that use
thread-local memory. Worker local tensors are generally small (on the order of hundreds of bytes) and should fit in
register space on device or L1 cache.

An OVL worker local tensor is defined by its shape and type and can be initialized to either
all zeros or all ones using the `zeros()` or `ones()` functions.

.. code-block:: python

    import opveclib as ovl
    # declare a 1x3 worker local tensor of type float32, initialized to all zeros
    a = ovl.zeros((1, 3), ovl.float32)
    for col in ovl.arange(3):
        a[col] = 0.9


Gradients
~~~~~~~~~

An OVL gradient operator is a special type of operator. It is defined by creating a python gradient function and
decorating it with the ``operator()`` decorator as well as the ``gradient()`` decorator. The arguement to the
gradient decorator function is the name of the operator it is the gradient for. The gradient function
must compute the gradients with respect to each of the operators' inputs, given gradients with respect to the operators'
outputs. Its arguments are the operator input tensors as well as the output gradient tensors from the next higher
layer in the case of back propagation through a neural network.

Below is a simple operator that computes the function

.. math::

      z_{i} = x_{i}^2  + 3.0 * y_{i}

and its gradient, and tests the gradient against tensorflow

.. testcode::

    import numpy as np
    import opveclib as ovl
    import tensorflow as tf

    @ovl.operator()
    def func(x, y):
        assert(x.shape == y.shape)

        pos = ovl.position_in(x.shape)
        z = ovl.output_like(x)

        # compute the function
        z[pos] = x[pos] * x[pos] + 3.0 * y[pos]
        return z


    @ovl.gradient(func)
    @ovl.operator()
    def func_grad(x, y, dz):
        assert(x.shape == y.shape)
        assert(x.shape == dz.shape)

        pos = ovl.position_in(x.shape)
        dx = ovl.output_like(x)
        dy = ovl.output_like(y)

        # compute the gradients w.r.t each input
        dx[pos] = 2.0 * x[pos] * dz[pos]
        dy[pos] = 3.0 * dz[pos]
        return dx, dy

    # define some input values
    a = tf.constant(np.arange(5.0))
    b = tf.constant(np.arange(5.0))
    grad_above = tf.constant(np.random.random(5))

    # run the operator in tensorflow
    with tf.Session() as s:
        func_ovl = ovl.as_tensorflow(func(a, b))

        # define a tensorflow reference operation
        func_tf = a * a + 3.0 * b

        # register the gradients of the ovl operator and the refenence in tensorflow
        func_grad_ovl = tf.gradients(func_ovl, [a, b], grad_ys=grad_above)
        func_grad_tf = tf.gradients(func_tf, [a, b], grad_ys=grad_above)

        # evaluate the outputs
        func_grad_ovl, func_grad_tf = s.run([func_grad_ovl, func_grad_tf])
        assert np.all(np.equal(func_grad_ovl, func_grad_tf))



Operator Merging
~~~~~~~~~~~~~~~

Typical modern GPUs are able to perform arithmetic operations faster than transfers of a data value in or out of host memory,
and are thus memory bandwidth limited. If 2 operators that are performed in sequence can be merged into a single operator that
shares data values in device memory, performance can be improved. In an application using multiple OVL operators in sequence,
the OVL optimizer will automatically merge any operators that it can. 2 OVL operators in sequence, from and to, will be merged if:

    1) the workgroup shape of both operators is the same
    2) the write indexing pattern of the from operator matches the read indexing pattern of the to operator
    3) the tensor shape of the output of the from operator matches the tensor shape of the input to the to operator

Optimization by automatically merging OVL operators is determined by the opt_level flag that is passed to the ``evaluate``
and the ``as_tensorflow`` functions. Merging is performed when opt_level >= 3. This is the default optimization level.
When operators are merged, their gradient operators will also be merged when possible.