/* Copyright 2016 Hewlett Packard Enterprise Development LP

 Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 the License. You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
 on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
 the specific language governing permissions and limitations under the License.*/

#include "dynamiclibop.h"
#include <vector>
#include <memory>
#include <iostream>

// turn off c++ name mangling
#define ADDCPU_EXPORT extern "C"

// does not help...
//#define EIGEN_USE_NONBLOCKING_THREAD_POOL

// worker functions
void Add2CPUWork(const float *in0, const float *in1, float* out, int64_t len,
                 uint32_t block_size, uint32_t thread_index) {
    int64_t begin = thread_index * block_size;
    int64_t end = begin + block_size;
    if (end > len) end = len;
    for (int64_t i = begin; i < end; i++ ) {
		out[i] = in0[i] + in1[i];
	}
}

void AddFDFCPUWork(const float *in0, const double *in1, const float* in2, float* out, int64_t len,
                   uint32_t block_size, uint32_t thread_index) {
    int64_t begin = thread_index * block_size;
    int64_t end = begin + block_size;
    if (end > len) end = len;
    for (int64_t i = begin; i < end; i++ ) {
		out[i] = in0[i] + in1[i] + in2[i];
	}
}

void SumSqCPUWork(const float *in0, const double *in1, float* out0, float* out1, int64_t len,
                 uint32_t block_size, uint32_t thread_index) {
    int64_t begin = thread_index * block_size;
    int64_t end = begin + block_size;
    if (end > len) end = len;
    for (int64_t i = begin; i < end; i++ ) {
		out0[i] = in0[i] + in1[i];
		out1[i] = out0[i] * out0[i];
	}
}

// CPU functions to be called by the TF dynamic_lib_addgpu_test.py

ADDCPU_EXPORT
void add2float(std::vector<std::shared_ptr<const InputParameter>> inputs,
		      std::vector<std::shared_ptr<OutputParameter>> outputs,
		      int num_threads, int thread_index, uint16_t *err) {
	if (inputs.size() != 2) {
	    *err = 1;
	    return;
	}
	if (outputs.size() != 1) {
	    *err = 1;
	    return;
	}

	float *out = outputs[0]->get<float>();
	const float *in0 = inputs[0]->get<float>();
	const float *in1 = inputs[1]->get<float>();
	int64_t len = inputs[0]->length();

	// Make ParallelFor use as many threads as possible.
	// based on https://github.com/tensorflow/tensorflow/blob/r0.9/tensorflow/core/lib/core/threadpool_test.cc
//    int64_t kHugeCost = 1 << 30;


	// bind the first 3 parameters to the input and output arrays. begin and end parameters are
	// placeholders that come from the ParallelFor parameters when it gets called
//	using namespace std::placeholders;    // adds visibility of _1, _2, _3,...
//	auto fn_work = std::bind(Add2CPUWork, in0, in1, out, _1, _2);
//	fn_work(0, len);
    // this fails the assertion in https://github.com/tensorflow/tensorflow/blob/r0.9/tensorflow/core/lib/core/threadpool.cc
    // at line 90 because it seems TF library was not compiled with EIGEN_USE_NONBLOCKING_THREAD_POOL
//	thread_pool->ParallelFor(len, kHugeCost, fn_work);

    uint32_t block_size = len / num_threads;
    if(len % num_threads > 0) block_size += 1;
    return Add2CPUWork(in0, in1, out, len, block_size, thread_index);
}

ADDCPU_EXPORT
void addFloatDoubleFloat(std::vector<std::shared_ptr<const InputParameter>> inputs,
	      std::vector<std::shared_ptr<OutputParameter>> outputs,
	      int num_threads, int thread_index, uint16_t *err) {
	if (inputs.size() != 3) {
	    *err = 1;
	    return;
	}

	if (outputs.size() != 1) {
	    *err = 1;
	    return;
	}

	int64_t len = inputs[0]->length();
	float *out = outputs[0]->get<float>();
	const float *in0 = inputs[0]->get<float>();
	const double *in1 = inputs[1]->get<double>();
	const float *in2 = inputs[2]->get<float>();

	uint32_t block_size = len / num_threads;
    if(len % num_threads > 0) block_size += 1;
    return AddFDFCPUWork(in0, in1, in2, out, len, block_size, thread_index);
}

ADDCPU_EXPORT
void sumAndSq(std::vector<std::shared_ptr<const InputParameter>> inputs,
	      std::vector<std::shared_ptr<OutputParameter>> outputs,
	      int num_threads, int thread_index, uint16_t *err) {
	if (inputs.size() != 2) {
	    *err = 1;
	    return;
	}
	if (outputs.size() != 2) {
	    *err = 1;
	    return;
	}

	int64_t len = inputs[0]->length();
	float *out0 = outputs[0]->get<float>();
	float *out1 = outputs[1]->get<float>();
	const float *in0 = inputs[0]->get<float>();
	const double *in1 = inputs[1]->get<double>();

	uint32_t block_size = len / num_threads;
    if(len % num_threads > 0) block_size += 1;
    return SumSqCPUWork(in0, in1, out0, out1, len, block_size, thread_index);
}
