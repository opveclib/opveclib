/* Copyright 2016 Hewlett Packard Enterprise Development LP

 Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 the License. You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
 on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
 the specific language governing permissions and limitations under the License.*/

#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <chrono>
#include <vector>
#include <thread>
#include "dynamiclibop.h"
#include "language_dtype.h"

#include "tensorflow/core/platform/env.h"
#include "threadpool.h"

typedef uint16_t
        (*C_FUNPTR)(std::vector<std::shared_ptr<const InputParameter>> inputs,
                    std::vector<std::shared_ptr<OutputParameter>> outputs,
                    int num_threads, int thread_idx, uint16_t *err);

struct TensorParam {
    void* data;
    opveclib::DType dtype;
    size_t len;
};

// Function which can run the fxxx_generic_cpp function from the
// operator generated library for testing
extern "C"
int32_t testCOperator(const char *opLibPath, const char *opFuncName,
                 const TensorParam* testInputs, const size_t numInputs,
                 TensorParam* testOutputs, const size_t numOutputs,
                 double * executionTimeMilliseconds,
                 const size_t profiling_iterations ) {
    // Build the tensor input parameter list
    std::vector<std::shared_ptr<const InputParameter>> inputs;
    inputs.reserve(numInputs);
    for (size_t i = 0; i < numInputs; ++i) {
        size_t N = testInputs[i].len;
        switch (testInputs[i].dtype) {
            case (opveclib::DType::FLOAT32): {
                inputs.emplace_back(
                    new TypedInput<float>(static_cast<const float*>(testInputs[i].data), N));
                break;
            }
            case (opveclib::DType::FLOAT64): {
                inputs.emplace_back(
                    new TypedInput<double>(static_cast<const double*>(testInputs[i].data), N));
                break;
            }
            case (opveclib::DType::INT8): {
                inputs.emplace_back(
                    new TypedInput<int8_t>(static_cast<const int8_t*>(testInputs[i].data), N));
                break;
            }
            case (opveclib::DType::INT16): {
                inputs.emplace_back(
                    new TypedInput<int16_t>(static_cast<const int16_t*>(testInputs[i].data), N));
                break;
            }
            case (opveclib::DType::INT32): {
                inputs.emplace_back(
                    new TypedInput<int32_t>(static_cast<const int32_t*>(testInputs[i].data), N));
                break;
            }
            case (opveclib::DType::INT64): {
                inputs.emplace_back(
                    new TypedInput<int64_t>(static_cast<const int64_t*>(testInputs[i].data), N));
                break;
            }
            case (opveclib::DType::UINT8): {
                inputs.emplace_back(
                    new TypedInput<uint8_t>(static_cast<const uint8_t*>(testInputs[i].data), N));
                break;
            }
            case (opveclib::DType::UINT16): {
                inputs.emplace_back(
                    new TypedInput<uint16_t>(static_cast<const uint16_t*>(testInputs[i].data), N));
                break;
            }
            case (opveclib::DType::UINT32): {
                inputs.emplace_back(
                    new TypedInput<uint32_t>(static_cast<const uint32_t*>(testInputs[i].data), N));
                break;
            }
            case (opveclib::DType::UINT64): {
                inputs.emplace_back(
                    new TypedInput<uint64_t>(static_cast<const uint64_t*>(testInputs[i].data), N));
                break;
            }
            default:
                std::cerr << "***ERROR - unsupported input type. " << testInputs[i].dtype << '\n';
                return 1;
          }
    }

    // Build the output tensor parameter list
    // initialize all outputs to 0xFE in order to catch errors where outputs
    // are not completely filled
    std::vector<std::shared_ptr<OutputParameter>> outputs;
    outputs.reserve(numOutputs);
    for (uint32_t i = 0; i < numOutputs; ++i) {
        size_t N = testOutputs[i].len;
        switch (testOutputs[i].dtype) {
            case (opveclib::DType::FLOAT32): {
                memset(testOutputs[i].data, 0xFE, N*sizeof(float));
                outputs.emplace_back(new TypedOutput<float>(
                               static_cast<float*>(testOutputs[i].data), N));
                break;
            }
            case (opveclib::DType::FLOAT64): {
                memset(testOutputs[i].data, 0xFE, N*sizeof(double));
                outputs.emplace_back(new TypedOutput<double>(
                               static_cast<double*>(testOutputs[i].data), N));
                break;
            }
            case (opveclib::DType::INT8): {
                memset(testOutputs[i].data, 0xFE, N*sizeof(int8_t));
                outputs.emplace_back(
                     new TypedOutput<int8_t>(static_cast<int8_t*>(testOutputs[i].data), N));
                break;
            }
            case (opveclib::DType::INT16): {
                memset(testOutputs[i].data, 0xFE, N*sizeof(int16_t));
                outputs.emplace_back(
                     new TypedOutput<int16_t>(static_cast<int16_t*>(testOutputs[i].data), N));
                break;
            }
            case (opveclib::DType::INT32): {
                memset(testOutputs[i].data, 0xFE, N*sizeof(int32_t));
                outputs.emplace_back(
                     new TypedOutput<int32_t>(static_cast<int32_t*>(testOutputs[i].data), N));
                break;
            }
            case (opveclib::DType::INT64): {
                memset(testOutputs[i].data, 0xFE, N*sizeof(int64_t));
                outputs.emplace_back(
                     new TypedOutput<int64_t>(static_cast<int64_t*>(testOutputs[i].data), N));
                break;
            }
            case (opveclib::DType::UINT8): {
                memset(testOutputs[i].data, 0xFE, N*sizeof(uint8_t));
                outputs.emplace_back(
                     new TypedOutput<uint8_t>(static_cast<uint8_t*>(testOutputs[i].data), N));
                break;
            }
            case (opveclib::DType::UINT16): {
                memset(testOutputs[i].data, 0xFE, N*sizeof(uint16_t));
                outputs.emplace_back(
                     new TypedOutput<uint16_t>(static_cast<uint16_t*>(testOutputs[i].data), N));
                break;
            }
            case (opveclib::DType::UINT32): {
                memset(testOutputs[i].data, 0xFE, N*sizeof(uint32_t));
                outputs.emplace_back(
                     new TypedOutput<uint32_t>(static_cast<uint32_t*>(testOutputs[i].data), N));
                break;
            }
            case (opveclib::DType::UINT64): {
                memset(testOutputs[i].data, 0xFE, N*sizeof(uint64_t));
                outputs.emplace_back(
                     new TypedOutput<uint64_t>(static_cast<uint64_t*>(testOutputs[i].data), N));
                break;
            }
            default:
                std::cerr << "***ERROR - unsupported output type. " << testOutputs[i].dtype << '\n';
                return 1;
         }
    }

    // load the operator library
//    std::cout << "loading function " <<  opFuncName << '\n';
//    std::cout << "from " <<  opLibPath << '\n';
    static_assert(sizeof(void *) == sizeof(void (*)(void)),
                      "object pointer and function pointer sizes must equal");
    void *handle = dlopen(opLibPath, RTLD_LAZY);
    if (handle == nullptr) {
        std::cerr << "***ERROR - Unable to find operator library " << opLibPath << '\n';
        return 1;
    }

    // load the function and cast it from void* to a function pointer
    void *f = dlsym(handle, opFuncName);
    C_FUNPTR func_ = reinterpret_cast<C_FUNPTR>(f);
    if (handle == nullptr) {
        std::cerr << "***ERROR - Unable to find operator function " << opFuncName << '\n';
        return 1;
    }

    // create the threadpool
    unsigned int num_threads = std::thread::hardware_concurrency();
//    std::cout << "num_threads: " + std::to_string(num_threads) << '\n';

    // call the test library function
    // time the execution in milliseconds
    uint16_t err = 0;
    for (size_t profiling_iter = 0; profiling_iter < profiling_iterations; profiling_iter++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        // ThreadPool destructor is what joins all the threads and waits for completion, so we
        // scope it as a local variable and when it goes out of scope, all the work is done and result
        // can be used
        {
            tensorflow::thread::ThreadPool thread_pool(tensorflow::Env::Default(), "test", num_threads);
            for (int thread = 0; thread < num_threads; thread++) {
                auto fn_work = std::bind(func_, inputs, outputs, num_threads, thread, &err);
                thread_pool.Schedule(fn_work);
            }
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dt_dur = t2 - t1;
        executionTimeMilliseconds[profiling_iter] = dt_dur.count();
    }

    if (err != 0)
        std::cerr << "***ERROR - Generated operator function execution error code: "
                  <<  err << '\n';

    return err;
}
