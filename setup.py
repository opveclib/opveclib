# Copyright 2016 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
# the specific language governing permissions and limitations under the License.

from setuptools import setup
from opveclib.local import version

setup(
    name='opveclib',
    version=version,
    packages=['opveclib', 'opveclib.test', 'opveclib.examples', 'opveclib.stdops'],
    install_requires=['numpy >= 1.11.0', 'protobuf >= 3.0.0', 'tensorflow >=  0.11.0', 'six >= 1.10.0'],
    package_data={
        'opveclib': ['dynamiclibop.h', 'dynamiclibop.cc',
                     'testcop.cc', 'testcudaop.cc',
                     'language.proto',
                     'test/addcpu.cpp', 'test/addgpu.cu',
                     'threadpool.h']
    },
    test_suite='nose2.collector.collector',
    license='Apache 2.0',
    description='Operator Vectorization Library',
    long_description='',
    url='https://github.com/opveclib/opveclib/',
    author='Hewlett Packard Labs',
    author_email='',
)
