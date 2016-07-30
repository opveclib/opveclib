Contributor's Guide
===================

Contribution Policy
-------------------

* Do not use a logging level above debug - library should be silent unless there is an error raised, or the user is
   explicitly capturing debug level logs.
* Do not use print statements. All debugging output should be made to the library's logger.
* All functionality must be tested.

Contributor License Agreement
-----------------------------

TODO: link to CLA here

Developers
-------------

If you are an opveclib developer and need to make changes to language.proto, you will need to install the protoc compiler.

.. code-block:: console

    sudo apt-get install protobuf-compiler

Note, this will install version 2.5 of protoc, which is only compatible with python 2. If you wish to use python 3,
you will need version 3.0.0-beta-2 of protoc. It can be found and installed from here:
https://github.com/google/protobuf/releases/

Local installation
------------------

Install python setuptools:

On Ubuntu:

.. code-block:: console

    sudo apt-get install python-setuptools

For other OS installations see https://pypi.python.org/pypi/setuptools#installation-instructions.

To install a local version of opveclib under your home directory directly from your local source:

.. code-block:: console

    python setup.py install --user

Testing
-------
Individual tests can be run on your local source from the root opveclib directory as follows:

.. code-block:: console

    nose2 opveclib.test.test_clip
    nose2 opveclib.examples.test_accumulate

etc.

Making Documentation
--------------------

To build the docs, you must first install Sphinx >= 1.4.1 with alabaster >= 0.7.8:

.. code-block:: console

    pip install -U sphinx

Properly rendering embedded math with mathjax requires installing the appropriate latex packages:

.. code-block:: console

    sudo apt-get install texlive-latex-base, texlive-latex-extra

The docs are built by navigating to the documentation directory and using make:

.. code-block:: console

    make html


This will output HTML to ``documentation/_build`` which can then be examined locally or published.


IDE configuration
-----------------

To run the local unit tests from within PyCharm, you have to set PyCharm to use nosetests as the default test runner.

In Pycharm choose File->Settings->Tools->Python Integrated tools.
Choose nosetests from the drop-down as the Defualt Test Runner.
Now if you right-click on a test script, you should see the option "Run nosetests in..."

Note: PyCharm sometimes has issues with stale cached settings. If you don't see nosetests as a run option, you may have
to restart PyCharm or as a last resort, delete the .idea directory in your base directory.

