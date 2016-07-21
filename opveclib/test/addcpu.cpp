/* Copyright 2016 Hewlett Packard Enterprise Development LP

 Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 the License. You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
 on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
 the specific language governing permissions and limitations under the License.*/

#include "dynamiclibop.h"
#include "threadpool.h"
#include <vector>
#include <memory>
#include <iostream>

// turn off c++ name mangling
#define ADDCPU_EXPORT extern "C"

//#define EIGEN_USE_NONBLOCKING_THREAD_POOL

// worker functions
void Add2CPUWork(const float *in0, const float *in1, float* out, int64_t index) {
//    for (int64_t i = begin; i < end; i++ ) {
		out[index] = in0[index] + in1[index];
//	}
}

// CPU functions to be called by the TF dynamic_lib_addgpu_test.py

ADDCPU_EXPORT
int add2float(std::vector<std::shared_ptr<const InputParameter>> inputs,
		      std::vector<std::shared_ptr<OutputParameter>> outputs, tensorflow::thread::ThreadPool *thread_pool) {
	if (inputs.size() != 2) return 1;
	if (outputs.size() != 1) return 1;

	float *out = outputs[0]->get<float>();
	const float *in0 = inputs[0]->get<float>();
	const float *in1 = inputs[1]->get<float>();
	int64_t len = inputs[0]->length();

	// Make ParallelFor use as many threads as possible.
//    int64_t kHugeCost = 1 << 30;
	std::cout << "*** Add2CPUWork - threads:  " + std::to_string(thread_pool->NumThreads()) << std::endl;

//	using namespace std::placeholders;    // adds visibility of _1, _2, _3,...
	// bind the first 3 parameters to the input and output arrays
//	auto fn_work = std::bind(Add2CPUWork, in0, in1, out, _1, _2);
//	fn_work(0, len);
//	thread_pool->ParallelFor(len, kHugeCost, fn_work);
    for (int64_t i = 0; i < len; i++ ) {
        auto fn_work = std::bind(Add2CPUWork, in0, in1, out, i);
        thread_pool->Schedule(fn_work);
    }

	return 0;
}

ADDCPU_EXPORT
int addFloatDoubleFloat(std::vector<std::shared_ptr<const InputParameter>> inputs,
	      std::vector<std::shared_ptr<OutputParameter>> outputs, const tensorflow::thread::ThreadPool *thread_pool) {
	if (inputs.size() != 3) return 1;
	if (outputs.size() != 1) return 1;

	int64_t N = inputs[0]->length();
	for (int i = 0; i < N; i++ ) {
		outputs[0]->setValue<float>(i, *(inputs[0]->get<float>(i)) + *(inputs[1]->get<double>(i))
				+ *(inputs[2]->get<float>(i)));
	}
	return 0;
}

ADDCPU_EXPORT
int sumAndSq(std::vector<std::shared_ptr<const InputParameter>> inputs,
	      std::vector<std::shared_ptr<OutputParameter>> outputs, const tensorflow::thread::ThreadPool *thread_pool) {
	if (inputs.size() != 2) return 1;
	if (outputs.size() != 2) return 1;

	int64_t N = inputs[0]->length();
	for (int i = 0; i < N; i++ ) {
		outputs[0]->setValue<float>(i, *(inputs[0]->get<float>(i)) + *(inputs[1]->get<double>(i)));
		outputs[1]->setValue<float>(i, *(outputs[0]->get<float>(i)) * *(outputs[0]->get<float>(i)));
	}
	return 0;
}
