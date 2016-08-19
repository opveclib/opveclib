#!/bin/bash
# Copyright 2016 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
# the specific language governing permissions and limitations under the License.

# Run all unit Tests for opveclib on python 2 and 3. Stop after first failure
nose2-2.7 -F opveclib.test --verbose
if [ ! $? -eq 0 ]; then
  echo "unit tests failed on Python 2"
  exit 1
fi
nose2-2.7 -F opveclib.stdops --verbose
if [ ! $? -eq 0 ]; then
  echo "unit tests failed on Python 2"
  exit 1
fi
nose2-2.7 -F opveclib.examples --verbose
if [ ! $? -eq 0 ]; then
  echo "example tests failed on Python 2"
  exit 1
fi
# now run in python 3
nose2-3.4 -F opveclib.test --verbose
if [ ! $? -eq 0 ]; then
  echo "unit tests failed on Python 3"
  exit 1
fi
nose2-3.4 -F opveclib.stdops --verbose
if [ ! $? -eq 0 ]; then
  echo "unit tests failed on Python 3"
  exit 1
fi
nose2-3.4 -F opveclib.examples --verbose
if [ ! $? -eq 0 ]; then
  echo "example tests failed on Python 3"
  exit 1
fi

# run doctest
cd documentation
make doctest
if [ ! $? -eq 0 ]; then
  echo "doctest failed"
  exit 1
fi