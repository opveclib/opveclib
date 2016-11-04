.. _contributor-label:

Contributor's Guide
===================

Contributor License Agreement
-----------------------------

We would love to take your contributions. To contribute to opveclib, you need to fill out a Contributor License Agreement (CLA).

You can download the CLA :download:`here <../HPE_CCLA.pdf>`.

Fill it out and email the completed agreement to the address in the agreement.


Contribution Policy
-------------------

* All functionality must be fully documented and tested. Doctest examples should be included in all
  documentation.
* Follow the PEP-8 style guide.
* Do not use print statements. All debugging output should be made to the library's logger.
* Library should be silent unless there is an error raised, or the user is
  explicitly capturing debug or info level logs.
* Use logging level info judiciously for confirmation that the library is functioning correctly.
* Use logging level debug for detailed information needed for debugging when things go wrong.

Versioning
----------

Code is versioned according to the `semantic versioning 2.0.0 <http://semver.org/spec/v2.0.0.html>`_ spec.
The complete public API is specified :ref:`here <api>`.

Developers Notes
----------------

Protocol Buffers
~~~~~~~~~~~~~~~~

In the rare case that you need to make any changes to the language.proto file, you will also need to compile and
check in a new language_pb2.py file. For the generated file to work with python 3 you must install and use a
version of the protoc compiler >= 3.0.0-beta-2. The compiler can be found and installed from:
https://github.com/google/protobuf/releases/

Making Documentation
~~~~~~~~~~~~~~~~~~~~

To build the docs, you must first install Sphinx >= 1.4.1 with alabaster >= 0.7.8:

.. code-block:: console

    pip install -U sphinx

Properly rendering embedded math with mathjax requires installing the appropriate latex packages:

.. code-block:: console

    sudo apt-get install texlive-latex-base, texlive-latex-extra

The docs are built by navigating to the ``documentation`` directory and using make:

.. code-block:: console

    make clean
    make html


This will output HTML to ``documentation/_build`` which can then be examined locally.

All docs containing code snippets must be tested with doctest. All doctests must pass before making a contribution by
running the following command from the ``documentation`` directory:

.. code-block:: console

    make doctest