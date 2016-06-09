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

# run opveclib python 2 unit tests. stop after first failure
# have to run the ones using tensorflow in a separate process as
# TF can't handle a cuda runtime api call that it did not make in its process
nose2-2.7 -F opveclib.test_tensorflow
if [ ! $? -eq 0 ]; then
  echo "tensorflow integration tests failed"
  exit 1
fi
nose2-2.7 -F opveclib.examples.tensorflow_clustering opveclib.examples
if [ ! $? -eq 0 ]; then
  echo "example tests failed"
  exit 1
fi
nose2-2.7 -F opveclib.test

# now run in python 3
nose2-3.4 -F opveclib.test
