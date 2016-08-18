/* Copyright 2016 Hewlett Packard Enterprise Development LP

 Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 the License. You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
 on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
 the specific language governing permissions and limitations under the License.*/

#include <cxxabi.h>
#include <cuda.h>
#include <dlfcn.h>
#include <iostream>
#include <string>
#include <memory>
#include <chrono>
#include <vector>
#include "dynamiclibop.h"
#include "language_dtype.h"

#define CUDA_RUNTIME_SAFE_CALL(x)                                     \
    do {                                                          \
        cudaError_t result = x;                                   \
        if (result != cudaSuccess) {                              \
            const char *msg =  cudaGetErrorString(result);        \
            std::cerr << "\nerror: " #x " failed with error "     \
                      << msg << '\n';                             \
            exit(1);                                              \
        }                                                         \
    } while (0)

#define CUDA_DRIVER_SAFE_CALL(x)                                  \
    do {                                                          \
        CUresult result = x;                                      \
        if (result != CUDA_SUCCESS) {                             \
            const char *msg;                                      \
            cuGetErrorString (result, &msg );                     \
            std::cerr << "\nerror: " #x " failed with error "     \
                      << msg << '\n';                             \
            exit(1);                                              \
        }                                                         \
    } while (0)

typedef uint16_t
        (*CUDA_FUNPTR)
                  (std::vector<std::shared_ptr<const InputParameter>> inputs,
                  std::vector<std::shared_ptr<OutputParameter>> outputs,
                  cudaStream_t stream,
                  uint16_t cuda_threads_per_block, uint16_t *err);

struct TensorParam {
    void* data;
    opveclib::DType dtype;
    size_t len;
};

// Function which can run the fxxx_generic_cuda function from the
// operator generated library for testing
extern "C"
int32_t testCUDAOperator(const char *opLibPath, const char *opFuncName,
                 const TensorParam* testInputs, const size_t numInputs,
                 TensorParam* testOutputs, const size_t numOutputs,
                 const uint16_t cuda_threads_per_block,
                 double * executionTimeMilliseconds,
                 const size_t profiling_iterations) {

    // initialize the CUDA driver API
    CUDA_DRIVER_SAFE_CALL(cuInit(0));
    // create a separate context for this test as if we run any tensorflow tests in the same process they will fail
    // as TF uses the StreamEexecutor api which is not compatible with any other CUDA calls
    // see https://github.com/tensorflow/tensorflow/issues/2210
    CUcontext ctx;
    CUDA_DRIVER_SAFE_CALL(cuCtxCreate(&ctx, 0, 0));
    CUDA_DRIVER_SAFE_CALL(cuCtxPushCurrent(ctx));

    // create the CUDA stream to run the test
    cudaStream_t stream1;
    CUDA_RUNTIME_SAFE_CALL(cudaStreamCreate(&stream1));

    // Build the tensor input parameter list
    std::vector<std::shared_ptr<const InputParameter>> inputs;
    inputs.reserve(numInputs);
    void* d_inputs[numInputs];
    for (size_t i = 0; i < numInputs; ++i) {
        size_t N = testInputs[i].len;
        switch (testInputs[i].dtype) {
            case (opveclib::DType::FLOAT32): {
                CUDA_RUNTIME_SAFE_CALL(cudaMalloc(&d_inputs[i], N *sizeof(float)));
                CUDA_RUNTIME_SAFE_CALL(cudaMemcpyAsync(d_inputs[i], testInputs[i].data,
                             N *sizeof(float),
                             cudaMemcpyHostToDevice, stream1));
                inputs.emplace_back(
                     new TypedInput<float>(static_cast<const float*>(d_inputs[i]), N));
                break;
            }
            case (opveclib::DType::FLOAT64): {
                CUDA_RUNTIME_SAFE_CALL(cudaMalloc(&d_inputs[i], N*sizeof(double)));
                CUDA_RUNTIME_SAFE_CALL(cudaMemcpyAsync(d_inputs[i], testInputs[i].data,
                               N*sizeof(double),
                               cudaMemcpyHostToDevice, stream1));
                inputs.emplace_back(
                     new TypedInput<double>(static_cast<const double*>(d_inputs[i]), N));
                break;
            }
            case (opveclib::DType::INT8): {
                CUDA_RUNTIME_SAFE_CALL(cudaMalloc(&d_inputs[i], N*sizeof(int8_t)));
                CUDA_RUNTIME_SAFE_CALL(cudaMemcpyAsync(d_inputs[i], testInputs[i].data,
                             N*sizeof(int8_t),
                             cudaMemcpyHostToDevice, stream1));
                inputs.emplace_back(
                     new TypedInput<int8_t>(static_cast<const int8_t*>(d_inputs[i]), N));
                break;
            }
            case (opveclib::DType::INT16): {
                CUDA_RUNTIME_SAFE_CALL(cudaMalloc(&d_inputs[i], N*sizeof(int16_t)));
                CUDA_RUNTIME_SAFE_CALL(cudaMemcpyAsync(d_inputs[i], testInputs[i].data,
                             N*sizeof(int16_t),
                             cudaMemcpyHostToDevice, stream1));
                inputs.emplace_back(
                     new TypedInput<int16_t>(static_cast<const int16_t*>(d_inputs[i]), N));
                break;
            }
            case (opveclib::DType::INT32): {
                CUDA_RUNTIME_SAFE_CALL(cudaMalloc(&d_inputs[i], N*sizeof(int32_t)));
                CUDA_RUNTIME_SAFE_CALL(cudaMemcpyAsync(d_inputs[i], testInputs[i].data,
                             N*sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream1));
                inputs.emplace_back(
                     new TypedInput<int32_t>(static_cast<const int32_t*>(d_inputs[i]), N));
                break;
            }
            case (opveclib::DType::INT64): {
                CUDA_RUNTIME_SAFE_CALL(cudaMalloc(&d_inputs[i], N*sizeof(int64_t)));
                CUDA_RUNTIME_SAFE_CALL(cudaMemcpyAsync(d_inputs[i], testInputs[i].data,
                             N*sizeof(int64_t),
                             cudaMemcpyHostToDevice, stream1));
                inputs.emplace_back(
                     new TypedInput<int64_t>(static_cast<const int64_t*>(d_inputs[i]), N));
                break;
            }
            case (opveclib::DType::UINT8): {
                CUDA_RUNTIME_SAFE_CALL(cudaMalloc(&d_inputs[i], N*sizeof(uint8_t)));
                CUDA_RUNTIME_SAFE_CALL(cudaMemcpyAsync(d_inputs[i], testInputs[i].data,
                             N*sizeof(uint8_t),
                             cudaMemcpyHostToDevice, stream1));
                inputs.emplace_back(
                     new TypedInput<uint8_t>(static_cast<const uint8_t*>(d_inputs[i]), N));
                break;
            }
            case (opveclib::DType::UINT16): {
                CUDA_RUNTIME_SAFE_CALL(cudaMalloc(&d_inputs[i], N*sizeof(uint16_t)));
                CUDA_RUNTIME_SAFE_CALL(cudaMemcpyAsync(d_inputs[i], testInputs[i].data,
                             N*sizeof(uint16_t),
                             cudaMemcpyHostToDevice, stream1));
                inputs.emplace_back(
                     new TypedInput<uint16_t>(static_cast<const uint16_t*>(d_inputs[i]), N));
                break;
            }
            case (opveclib::DType::UINT32): {
                CUDA_RUNTIME_SAFE_CALL(cudaMalloc(&d_inputs[i], N*sizeof(uint32_t)));
                CUDA_RUNTIME_SAFE_CALL(cudaMemcpyAsync(d_inputs[i], testInputs[i].data,
                             N*sizeof(uint32_t),
                             cudaMemcpyHostToDevice, stream1));
                inputs.emplace_back(
                     new TypedInput<uint32_t>(static_cast<const uint32_t*>(d_inputs[i]), N));
                break;
            }
            case (opveclib::DType::UINT64): {
                CUDA_RUNTIME_SAFE_CALL(cudaMalloc(&d_inputs[i], N*sizeof(uint64_t)));
                CUDA_RUNTIME_SAFE_CALL(cudaMemcpyAsync(d_inputs[i], testInputs[i].data,
                             N*sizeof(uint64_t),
                             cudaMemcpyHostToDevice, stream1));
                inputs.emplace_back(
                     new TypedInput<uint64_t>(static_cast<const uint64_t*>(d_inputs[i]), N));
                break;
            }
            default:
                std::cerr << "***ERROR - unsupported input type. "
                          << testInputs[i].dtype << '\n';
                return 1;
          }
    }

    // Build the output tensor parameter list
    // initialize all outputs to 0xFE in order to catch errors where outputs
    // are not completely filled
    std::vector<std::shared_ptr<OutputParameter>> outputs;
    outputs.reserve(numOutputs);
    void* d_outputs[numOutputs];
    for (uint32_t i = 0; i < numOutputs; ++i) {
        size_t N = testOutputs[i].len;
        switch (testOutputs[i].dtype) {
            case (opveclib::DType::FLOAT32): {
                CUDA_RUNTIME_SAFE_CALL(cudaMalloc(&d_outputs[i], N*sizeof(float)));
                CUDA_RUNTIME_SAFE_CALL(cudaMemset(d_outputs[i], 0xFE, N*sizeof(float)));
                outputs.emplace_back(new TypedOutput<float>(
                               static_cast<float*>(d_outputs[i]), N));
                break;
            }
            case (opveclib::DType::FLOAT64): {
                CUDA_RUNTIME_SAFE_CALL(cudaMalloc(&d_outputs[i], N*sizeof(double)));
                CUDA_RUNTIME_SAFE_CALL(cudaMemset(d_outputs[i], 0xFE, N*sizeof(double)));
                outputs.emplace_back(new TypedOutput<double>(
                               static_cast<double*>(d_outputs[i]), N));
                break;
            }
            case (opveclib::DType::INT8): {
                CUDA_RUNTIME_SAFE_CALL(cudaMalloc(&d_outputs[i], N*sizeof(int8_t)));
                CUDA_RUNTIME_SAFE_CALL(cudaMemset(d_outputs[i], 0xFE, N*sizeof(int8_t)));
                outputs.emplace_back(
                     new TypedOutput<int8_t>(static_cast<int8_t*>(d_outputs[i]), N));
                break;
            }
            case (opveclib::DType::INT16): {
                CUDA_RUNTIME_SAFE_CALL(cudaMalloc(&d_outputs[i], N*sizeof(int16_t)));
                CUDA_RUNTIME_SAFE_CALL(cudaMemset(d_outputs[i], 0xFE, N*sizeof(int16_t)));
                outputs.emplace_back(
                     new TypedOutput<int16_t>(static_cast<int16_t*>(d_outputs[i]), N));
                break;
            }
            case (opveclib::DType::INT32): {
                CUDA_RUNTIME_SAFE_CALL(cudaMalloc(&d_outputs[i], N*sizeof(int32_t)));
                CUDA_RUNTIME_SAFE_CALL(cudaMemset(d_outputs[i], 0xFE, N*sizeof(int32_t)));
                outputs.emplace_back(
                     new TypedOutput<int32_t>(static_cast<int32_t*>(d_outputs[i]), N));
                break;
            }
            case (opveclib::DType::INT64): {
                CUDA_RUNTIME_SAFE_CALL(cudaMalloc(&d_outputs[i], N*sizeof(int64_t)));
                CUDA_RUNTIME_SAFE_CALL(cudaMemset(d_outputs[i], 0xFE, N*sizeof(int64_t)));
                outputs.emplace_back(
                     new TypedOutput<int64_t>(static_cast<int64_t*>(d_outputs[i]), N));
                break;
            }
            case (opveclib::DType::UINT8): {
                CUDA_RUNTIME_SAFE_CALL(cudaMalloc(&d_outputs[i], N*sizeof(uint8_t)));
                CUDA_RUNTIME_SAFE_CALL(cudaMemset(d_outputs[i], 0xFE, N*sizeof(uint8_t)));
                outputs.emplace_back(
                     new TypedOutput<uint8_t>(static_cast<uint8_t*>(d_outputs[i]), N));
                break;
            }
            case (opveclib::DType::UINT16): {
                CUDA_RUNTIME_SAFE_CALL(cudaMalloc(&d_outputs[i], N*sizeof(uint16_t)));
                CUDA_RUNTIME_SAFE_CALL(cudaMemset(d_outputs[i], 0xFE, N*sizeof(uint16_t)));
                outputs.emplace_back(
                     new TypedOutput<uint16_t>(static_cast<uint16_t*>(d_outputs[i]), N));
                break;
            }
            case (opveclib::DType::UINT32): {
                CUDA_RUNTIME_SAFE_CALL(cudaMalloc(&d_outputs[i], N*sizeof(uint32_t)));
                CUDA_RUNTIME_SAFE_CALL(cudaMemset(d_outputs[i], 0xFE, N*sizeof(uint32_t)));
                outputs.emplace_back(
                     new TypedOutput<uint32_t>(static_cast<uint32_t*>(d_outputs[i]), N));
                break;
            }
            case (opveclib::DType::UINT64): {
                CUDA_RUNTIME_SAFE_CALL(cudaMalloc(&d_outputs[i], N*sizeof(uint64_t)));
                CUDA_RUNTIME_SAFE_CALL(cudaMemset(d_outputs[i], 0xFE, N*sizeof(uint64_t)));
                outputs.emplace_back(
                     new TypedOutput<uint64_t>(static_cast<uint64_t*>(d_outputs[i]), N));
                break;
            }
            default:
                std::cerr << "***ERROR - unsupported output type. "
                          << testOutputs[i].dtype << '\n';
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
        std::cerr << "***ERROR - Unable to find operator library "
                  << opLibPath << '\n';
        return 1;
    }

    // load the function and cast it from void* to a function pointer
    void *f = dlsym(handle, opFuncName);
    CUDA_FUNPTR func_ = reinterpret_cast<CUDA_FUNPTR>(f);
    if (handle == nullptr) {
        std::cerr << "***ERROR - Unable to find operator function "
                  << opFuncName << '\n';
        return 1;
    }

    // call the test library function
    // time the execution in milliseconds
    cudaDeviceSynchronize();
    uint16_t err = 0;
    for (size_t profiling_iter = 0; profiling_iter < profiling_iterations; profiling_iter++) {
        cudaStreamSynchronize(stream1);
        auto t1 = std::chrono::high_resolution_clock::now();
        func_(inputs, outputs, stream1, cuda_threads_per_block, &err);
        cudaStreamSynchronize(stream1);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dt_dur = t2 - t1;
        executionTimeMilliseconds[profiling_iter] = dt_dur.count();
    }

    if (err != 0) {
        std::cerr << "***ERROR - Generated operator function execution error code: "
                  <<  err << '\n';
    } else {
        // copy results back from device
        for (uint32_t i = 0; i < numOutputs; ++i) {
            size_t N = testOutputs[i].len;
            switch (testOutputs[i].dtype) {
                case (opveclib::DType::FLOAT32): {
                    CUDA_RUNTIME_SAFE_CALL(cudaMemcpyAsync(testOutputs[i].data, d_outputs[i],
                                   N*sizeof(float),
                                   cudaMemcpyDeviceToHost, stream1));
                break;
                }
                case (opveclib::DType::FLOAT64): {
                    CUDA_RUNTIME_SAFE_CALL(cudaMemcpyAsync(testOutputs[i].data, d_outputs[i],
                                   N*sizeof(double),
                                   cudaMemcpyDeviceToHost, stream1));
                break;
                }
                case (opveclib::DType::INT8): {
                    CUDA_RUNTIME_SAFE_CALL(cudaMemcpyAsync(testOutputs[i].data, d_outputs[i],
                                   N*sizeof(int8_t),
                                   cudaMemcpyDeviceToHost, stream1));
                    break;
                }
                case (opveclib::DType::INT16): {
                    CUDA_RUNTIME_SAFE_CALL(cudaMemcpyAsync(testOutputs[i].data, d_outputs[i],
                                   N*sizeof(int16_t),
                                   cudaMemcpyDeviceToHost, stream1));
                    break;
                }
                case (opveclib::DType::INT32): {
                    CUDA_RUNTIME_SAFE_CALL(cudaMemcpyAsync(testOutputs[i].data, d_outputs[i],
                                   N*sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, stream1));
                    break;
                }
                case (opveclib::DType::INT64): {
                    CUDA_RUNTIME_SAFE_CALL(cudaMemcpyAsync(testOutputs[i].data, d_outputs[i],
                                   N*sizeof(int64_t),
                                   cudaMemcpyDeviceToHost, stream1));
                    break;
                }
                case (opveclib::DType::UINT8): {
                    CUDA_RUNTIME_SAFE_CALL(cudaMemcpyAsync(testOutputs[i].data, d_outputs[i],
                                   N*sizeof(uint8_t),
                                   cudaMemcpyDeviceToHost, stream1));
                    break;
                }
                case (opveclib::DType::UINT16): {
                    CUDA_RUNTIME_SAFE_CALL(cudaMemcpyAsync(testOutputs[i].data, d_outputs[i],
                                   N*sizeof(uint16_t),
                                   cudaMemcpyDeviceToHost, stream1));
                    break;
                }
                case (opveclib::DType::UINT32): {
                    CUDA_RUNTIME_SAFE_CALL(cudaMemcpyAsync(testOutputs[i].data, d_outputs[i],
                                   N*sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, stream1));
                    break;
                }
                case (opveclib::DType::UINT64): {
                    CUDA_RUNTIME_SAFE_CALL(cudaMemcpyAsync(testOutputs[i].data, d_outputs[i],
                                   N*sizeof(uint64_t),
                                   cudaMemcpyDeviceToHost, stream1));
                    break;
                }
                default:
                    std::cerr << "***ERROR - unsupported output type. "
                              << testOutputs[i].dtype << '\n';
            }
        }
    }

    // clean up
    for (uint32_t i = 0; i < numInputs; ++i) {
        CUDA_RUNTIME_SAFE_CALL(cudaFree(d_inputs[i]));
    }
    for (uint32_t i = 0; i < numOutputs; ++i) {
        CUDA_RUNTIME_SAFE_CALL(cudaFree(d_outputs[i]));
    }
    CUDA_RUNTIME_SAFE_CALL(cudaStreamDestroy(stream1));

    // return to the old context
    CUcontext oldCtx;
    CUDA_DRIVER_SAFE_CALL(cuCtxPopCurrent(&oldCtx));
    CUDA_DRIVER_SAFE_CALL(cuCtxDestroy(ctx));
    return err;
}
