/* Copyright 2016 Hewlett Packard Enterprise Development LP

 Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 the License. You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
 on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
 the specific language governing permissions and limitations under the License.*/

#include "dynamiclibop.h"
#include <cuda.h>
#include <vector>
#include <iostream>
#include <memory>
#include <assert.h>

// turn off c++ name mangling
#define ADDGPU_EXPORT extern "C"

// GPU functions to be called by the TF dynamic_lib_addgpu_test.py
// code must be compilable by nvcc

// cuda kernels

__global__ void Add2GPUKernel(const float *in0, const float *in1,  float* out, int size) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_thread_count = gridDim.x * blockDim.x;

  int offset = thread_id;

  while (offset < size) {
    out[offset] = in0[offset] + in1[offset] + 1.0;
    offset += total_thread_count;
  }
}

__global__ void Add3GPUKernel(const float *in0, const double *in1,  const float *in2, float* out, int size) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_thread_count = gridDim.x * blockDim.x;

  int offset = thread_id;

  while (offset < size) {
    out[offset] = in0[offset] + in1[offset] + in2[offset] + 1.0;
    offset += total_thread_count;
  }
}

__global__ void SumSqGPUKernel(const float *in0, const double *in1,  float* out0, float *out1, int size) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_thread_count = gridDim.x * blockDim.x;

  int offset = thread_id;

  while (offset < size) {
    out0[offset] = in0[offset] + in1[offset] + 1.0;
    out1[offset] = out0[offset] * out0[offset] + 1.0;
    offset += total_thread_count;
  }
}

// dynamic library operators

ADDGPU_EXPORT
void add2float(std::vector<std::shared_ptr<const InputParameter>> inputs,
		      std::vector<std::shared_ptr<OutputParameter>> outputs, CUstream stream,
		      uint16_t threads_per_block, uint16_t *err) {
	if (inputs.size() != 2) { *err = 1; return; }
	if (outputs.size() != 1) { *err = 1; return; }

	float *out = outputs[0]->get<float>();
	const float *in0 = inputs[0]->get<float>();
	const float *in1 = inputs[1]->get<float>();
	int64_t len = inputs[0]->length();
	uint32_t num_blocks = len / threads_per_block;
	if(len % threads_per_block > 0) num_blocks += 1;

	std::cout << "*** launching Add2GPUKernel ***" << std::endl;
	Add2GPUKernel<<<num_blocks, threads_per_block, 0, stream>>>(in0, in1, out, len);
}

ADDGPU_EXPORT
void addFloatDoubleFloat(std::vector<std::shared_ptr<const InputParameter>> inputs,
		      std::vector<std::shared_ptr<OutputParameter>> outputs, CUstream stream,
		      uint16_t threads_per_block, uint16_t *err) {
	if (inputs.size() != 3) { *err = 1; return; }
	if (outputs.size() != 1) { *err = 1; return; }


	float *out = outputs[0]->get<float>();
	const float *in0 = inputs[0]->get<float>();
	const double *in1 = inputs[1]->get<double>();
	const float *in2 = inputs[2]->get<float>();
	int64_t len = inputs[0]->length();
	uint32_t num_blocks = len / threads_per_block;
	if(len % threads_per_block > 0) num_blocks += 1;

	std::cout << "*** launching Add3GPUKernel ***" << std::endl;
	Add3GPUKernel<<<num_blocks, threads_per_block, 0, stream>>>(in0, in1, in2, out, len);
}

ADDGPU_EXPORT
void sumAndSq(std::vector<std::shared_ptr<const InputParameter>> inputs,
		      std::vector<std::shared_ptr<OutputParameter>> outputs, CUstream stream,
		      uint16_t threads_per_block, uint16_t *err) {
	if (inputs.size() != 2) { *err = 1; return; }
	if (outputs.size() != 2) { *err = 1; return; }

	float *out0 = outputs[0]->get<float>();
	float *out1 = outputs[1]->get<float>();
	const float *in0 = inputs[0]->get<float>();
	const double *in1 = inputs[1]->get<double>();
	int64_t len = inputs[0]->length();
	uint32_t num_blocks = len / threads_per_block;
	if(len % threads_per_block > 0) num_blocks += 1;

	std::cout << "*** launching SumSqGPUKernel ***" << std::endl;
	SumSqGPUKernel<<<num_blocks, threads_per_block, 0, stream>>>(in0, in1, out0, out1, len);
}




