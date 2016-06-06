/* Copyright 2016 Hewlett Packard Enterprise Development LP

 Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 the License. You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
 on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
 the specific language governing permissions and limitations under the License.*/

#include "parameter.h"
#include <vector>
#include <memory>

// turn off c++ name mangling
#define ADDCPU_EXPORT extern "C"

// CPU functions to be called by the TF dynamic_lib_addgpu_test.py

ADDCPU_EXPORT
int add2float(std::vector<std::shared_ptr<const InputParameter>> inputs,
		      std::vector<std::shared_ptr<OutputParameter>> outputs) {
	if (inputs.size() != 2) return 1;
	if (outputs.size() != 1) return 1;

	int64_t N = inputs[0]->length();
	for (int i = 0; i < N; i++ ) {
		outputs[0]->setValue<float>(i, *(inputs[0]->get<float>(i)) + *(inputs[1]->get<float>(i)));
	}
	return 0;
}

ADDCPU_EXPORT
int addFloatDoubleFloat(std::vector<std::shared_ptr<const InputParameter>> inputs,
	      std::vector<std::shared_ptr<OutputParameter>> outputs) {
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
	      std::vector<std::shared_ptr<OutputParameter>> outputs) {
	if (inputs.size() != 2) return 1;
	if (outputs.size() != 2) return 1;

	int64_t N = inputs[0]->length();
	for (int i = 0; i < N; i++ ) {
		outputs[0]->setValue<float>(i, *(inputs[0]->get<float>(i)) + *(inputs[1]->get<double>(i)));
		outputs[1]->setValue<float>(i, *(outputs[0]->get<float>(i)) * *(outputs[0]->get<float>(i)));
	}
	return 0;
}
