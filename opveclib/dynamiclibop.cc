/* Copyright 2016 Hewlett Packard Enterprise Development LP

 Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 the License. You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
 on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
 the specific language governing permissions and limitations under the License.*/

// #include <cxxabi.h>
#include "dynamiclibop.h"
#include <dlfcn.h>
#include <string>
#include <memory>
#include <typeinfo>
#include <vector>
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/device_base.h"
#include "threadpool.h"

#if GOOGLE_CUDA

// this must be defined for the include file below to actually include anything
#define EIGEN_USE_GPU
#include <cuda.h>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#endif  // GOOGLE_CUDA

namespace tensorflow {

// Define the operator interface: inputs, outputs and parameters
// inputs and outputs are a list of tensors that can be either floats or doubles
// and all input tensors do not need to be the same type
REGISTER_OP("DynamicLib")
    .Attr("gpu_func_name: string")
    .Attr("gpu_lib_path: string")
    .Attr("cpu_func_name: string")
    .Attr("cpu_lib_path: string")
    .Attr("serialized_grad_dag: string")
    .Attr("grad_dag_arg_index: list(int)")
    .Attr("cuda_threads_per_block: int")
    .Attr("out_shapes: list(shape)")
    .Attr("in_types: list({float, double}) >= 0")
    .Attr("out_types: list({float, double})")
    .Input("inputs: in_types")
    .Output("outputs: out_types")
    .Doc(R"doc(call a dynamically generated library operation)doc");


typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// static std::string getDebugString(
//        const std::vector<std::shared_ptr<const InputParameter>> parameters) {
//    std::string str;
//    int     status;
//    char   *paramType;
//    for (uint32_t i = 0; i < parameters.size(); i++) {
//        const std::type_info  &ti = typeid(*(parameters[i]));
//        paramType = abi::__cxa_demangle(ti.name(), 0, 0, &status);
//        str.append(paramType + std::string(": ") +
//                   std::to_string(parameters[i]->length()) + ", ");
//    }
//    return str;
// }

// Class which will dynamically load and launch the generated operators
// on either CPU or GPU
template <typename Device>
class DynamicLibLaunch;

// Partial specialization for CPU
template <>
class DynamicLibLaunch<CPUDevice>  {
 public:
    typedef uint16_t
        (*FUNPTR)(std::vector<std::shared_ptr<const InputParameter>> inputs,
                  std::vector<std::shared_ptr<OutputParameter>> outputs,
                  int num_threads, int thread_idx, uint16_t *err);
    DynamicLibLaunch(OpKernelConstruction* context,
                     const string& cpu_func_name, const string& cpu_lib_path,
                     const string&, const string&,
                     const int ) {
        LOG(INFO) << "*** Standalone DynamicLibLaunch on CPU *****";

        // load the compiled op shared library
        static_assert(sizeof(void *) == sizeof(void (*)(void)),
                      "object pointer and function pointer sizes must equal");
        void *handle = dlopen(cpu_lib_path.c_str(), RTLD_LAZY);
        OP_REQUIRES(context, handle != nullptr,
            errors::NotFound("Unable to find DynamicLib library "
                             + cpu_lib_path));

        // load the function and cast it from void* to a function pointer
        void *f = dlsym(handle, cpu_func_name.c_str());
        func_ = reinterpret_cast<FUNPTR>(f);
        OP_REQUIRES(context, func_ != nullptr,
            errors::NotFound("Unable to find DynamicLib function "
                             + cpu_func_name));
    }

    void Run(OpKernelContext* context, const CPUDevice&,
             std::vector<std::shared_ptr<const InputParameter>> inputs,
             std::vector<std::shared_ptr<OutputParameter>> outputs) {
        const int num_threads = context->device()->tensorflow_cpu_worker_threads()->workers->NumThreads();
        uint16_t err = 0;
        // ThreadPool destructor is what joins all the threads and waits for completion, so we
        // scope it as a local variable and when it goes out of scope, all the work is done and result
        // can be used
        {
            thread::ThreadPool thread_pool(Env::Default(), "dynamiclibop", num_threads);
            for (int thread = 0; thread < num_threads; thread++) {
                auto fn_work = std::bind(func_, inputs, outputs, num_threads, thread, &err);
                thread_pool.Schedule(fn_work);
            }
        }
        OP_REQUIRES(context, err == 0,
            errors::InvalidArgument("External function execution error code: ",
                                    err));
    }

 private:
    FUNPTR func_;
};

#if GOOGLE_CUDA
// Partial specialization for GPU
template <>
class DynamicLibLaunch<GPUDevice>  {
 public:
    typedef uint16_t
        (*FUNPTR)(std::vector<std::shared_ptr<const InputParameter>> inputs,
                  std::vector<std::shared_ptr<OutputParameter>> outputs,
                  CUstream stream,
                  int cuda_threads_per_block, uint16_t *err);
    DynamicLibLaunch(OpKernelConstruction* context,
                     const string&, const string&,
                     const string& gpu_func_name, const string& gpu_lib_path,
                     const int cuda_threads_per_block) {
        LOG(INFO) << "*** Standalone DynamicLibLaunch on GPU *****";

        // load the compiled op shared library
        static_assert(sizeof(void *) == sizeof(void (*)(void)),
                  "object pointer and function pointer sizes must equal");
        void *handle = dlopen(gpu_lib_path.c_str(), RTLD_LAZY);
        OP_REQUIRES(context, handle != nullptr,
            errors::NotFound("Unable to find DynamicLib library "
                             + gpu_lib_path));

        // load the function and cast it from void* to a function pointer
        void *f = dlsym(handle, gpu_func_name.c_str());
        func_ = reinterpret_cast<FUNPTR>(f);
        OP_REQUIRES(context, func_ != nullptr,
            errors::NotFound("Unable to find DynamicLib function "
                             + gpu_func_name));

        cuda_threads_per_block_ = cuda_threads_per_block;
    }

    void Run(OpKernelContext* context, const GPUDevice& d,
             std::vector<std::shared_ptr<const InputParameter>> inputs,
             std::vector<std::shared_ptr<OutputParameter>> outputs) {
        // call the DynamicLib library functions
        uint16_t err = 0;
        func_(inputs, outputs, d.stream(),
                             cuda_threads_per_block_, &err);
        OP_REQUIRES(context, err == 0,
            errors::InvalidArgument("External function execution error code: ",
                                    err));
    }

 private:
    FUNPTR func_;
    int cuda_threads_per_block_;
};
#endif  // GOOGLE_CUDA

// General purpose tensorflow user operator class
// that allows us to run operators  generated and compiled independently
// by the Operator Vectorization Library.
// Parameters are a list of input tensors, list of output shapes, list of
// output types, the location of the DynamicLib shared libraries and the name
// of the DynamicLib operator.
// See tensorflow/python/kernel_tests/dynamic_lib_op_test.py for example usage
template <typename Device>
class DynamicLibOp : public OpKernel {
 public:
  explicit DynamicLibOp(OpKernelConstruction* context) : OpKernel(context) {
    // store the passed in parameters
    OP_REQUIRES_OK(context, context->GetAttr("cpu_func_name", &cpu_func_name_));
    OP_REQUIRES_OK(context, context->GetAttr("cpu_lib_path", &cpu_lib_path_));
    OP_REQUIRES_OK(context, context->GetAttr("gpu_func_name", &gpu_func_name_));
    OP_REQUIRES_OK(context, context->GetAttr("gpu_lib_path", &gpu_lib_path_));
    OP_REQUIRES_OK(context, context->GetAttr("cuda_threads_per_block",
                                             &cuda_threads_per_block_));
    OP_REQUIRES_OK(context, context->GetAttr("out_types", &out_types_));
    OP_REQUIRES_OK(context, context->GetAttr("out_shapes", &out_shapes_));

    launcher_ = std::unique_ptr<DynamicLibLaunch<Device>>(
                new DynamicLibLaunch<Device>(context, cpu_func_name_,
                                             cpu_lib_path_, gpu_func_name_,
                                             gpu_lib_path_,
                                             cuda_threads_per_block_));
  }

  // Function that is called when the output tensors of the operator
  // are evaluated
  void Compute(OpKernelContext* context) override {
//      LOG(INFO) << "*** computing DynamicLibOp ***";

      // Build the tensor input parameter list
      OpInputList input_list;
      context->input_list("inputs", &input_list);
      std::vector<std::shared_ptr<const InputParameter>> inputs;
      inputs.reserve(input_list.size());
      for (int32_t i = 0; i < input_list.size(); ++i) {
          const Tensor& cur_input = input_list[i];
          switch (cur_input.dtype()) {
            case (DT_FLOAT):
              inputs.emplace_back(
                     new TypedInput<float>(cur_input.flat<float>().data(),
                                           cur_input.NumElements()));
              break;
            case (DT_DOUBLE):
              inputs.emplace_back(
                     new TypedInput<double>(cur_input.flat<double>().data(),
                                            cur_input.NumElements()));
              break;
            default:
              OP_REQUIRES(context, false,
                          errors::InvalidArgument(
                          "Only float and double inputs are supported."));
              break;
            }
      }

      // Build the output tensor parameter list
      const uint32_t num_outputs = context->num_outputs();
      OP_REQUIRES(context, num_outputs == out_shapes_.size(),
                  errors::InvalidArgument(
                  "Output shapes inconsistent with output types"))
      OP_REQUIRES(context, num_outputs == out_types_.size(),
                  errors::InvalidArgument(
                  "Output types inconsistent num_outputs"))
      Tensor *output_tensor[num_outputs];
      std::vector<std::shared_ptr<OutputParameter>> outputs;
      outputs.reserve(num_outputs);
      for (uint32_t i = 0; i < num_outputs; ++i) {
          DataType cur_output_type = out_types_[i];
          TensorShape cur_shape = TensorShape(out_shapes_[i]);
          OP_REQUIRES_OK(context,
                         context->allocate_output(i,
                                              cur_shape, &output_tensor[i]));
          OP_REQUIRES(context, output_tensor[i]->dtype() == cur_output_type,
                       errors::InvalidArgument("Types inconsistent"))
          switch (cur_output_type) {
            case (DT_FLOAT):
                outputs.emplace_back(new TypedOutput<float>(
                               output_tensor[i]->template flat<float>().data(),
                               output_tensor[i]->NumElements()));
                break;
            case (DT_DOUBLE):
                outputs.emplace_back(new TypedOutput<double>(
                               output_tensor[i]->template flat<double>().data(),
                               output_tensor[i]->NumElements()));
                break;
            default:
                OP_REQUIRES(context, false,
                    errors::InvalidArgument(
                            "Only float and double outputs are supported."));
                break;
          }
      }

      // call the DynamicLib library function
      const Device& d = context->eigen_device<Device>();
      launcher_->Run(context, d, inputs, outputs);
  }

 private:
  string cpu_func_name_;
  string cpu_lib_path_;
  string gpu_func_name_;
  string gpu_lib_path_;
  int cuda_threads_per_block_;
  DataTypeVector out_types_;
  std::vector<TensorShapeProto> out_shapes_;
  std::unique_ptr<DynamicLibLaunch<Device>> launcher_;
};

// register the operator for each template type with tensorflow
// Note: this registration will cause the operator constructors to get called
// regardless of whether or not they are used in a tensorflow application
REGISTER_KERNEL_BUILDER(Name("DynamicLib")
    .Device(DEVICE_CPU),
    DynamicLibOp<CPUDevice>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("DynamicLib")
    .Device(DEVICE_GPU),
    DynamicLibOp<GPUDevice>);
#endif  // GOOGLE_CUDA
}  // namespace tensorflow

