import os
import sys


def add_rel_path():
    """
    Provide a simple way for adding the base source directory to the python path. This enables running tests and examples
    which reside in subdirectories of the source tree directly on the source library.
    """
    module_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, module_dir)
