/* Copyright 2016 Hewlett Packard Enterprise Development LP

 Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 the License. You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
 on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
 the specific language governing permissions and limitations under the License.*/

#ifndef DYNAMICLIBOP_H_
#define DYNAMICLIBOP_H_

#include <stdint.h>

// Classes to allow variable length lists of input and output tensors of varying
// types. Code must be compilable by nvcc if used for gpu operators
// Note: inputs and outputs must be separate types to preserve the const-ness
// of the underlying data of input tensors. Based from
// http://stackoverflow.com/questions/13980157/c-class-with-template-member-variable
class InputParameter {
 public:
    explicit InputParameter(int64_t len) : length_(len) {}
    // needed to make the class polymorphic so the dynamic_cast works
    virtual ~InputParameter() {}

    // forward declaration
    template<class T> const T* get(int64_t idx = 0) const;

    int64_t length() const { return length_; }

 private:
    int64_t length_;
};

template <typename T>
class TypedInput : public InputParameter {
 public:
    TypedInput(const T* rhs, int64_t len) : InputParameter(len), value_(rhs) {}

    const T* get(int64_t idx = 0) const { return (value_ + idx); }

 private:
    // pointer to underlying TF tensor data - we don't own this
    const T* value_;
};

class OutputParameter {
 public:
    explicit OutputParameter(int64_t len) : length_(len) {}
    // needed to make the class polymorphic so the dynamic_cast works
    virtual ~OutputParameter() {}

    // forward declarations
    template<class T>  T* get(int64_t idx = 0) const;
    template<class T, class U> void setValue(int64_t idx, const U& rhs);

    int64_t length() const { return length_; }

 private:
    int64_t length_;
};

template <typename T>
class TypedOutput : public OutputParameter {
 public:
    TypedOutput(T* rhs, int64_t len) : OutputParameter(len), value_(rhs) {}

    T* get(int64_t idx = 0) const { return (value_ + idx); }
    void setValue(int64_t idx, const T& rhs) { *(value_ + idx) = rhs; }

 private:
    // pointer to underlying TF tensor data - we don't own this
    T* value_;
};

template<class T>
const T* InputParameter::get(int64_t idx) const {
    return dynamic_cast<const TypedInput<T>&>(*this).get(idx);
}

template<class T>
T* OutputParameter::get(int64_t idx) const {
    return dynamic_cast<const TypedOutput<T>&>(*this).get(idx);
}

template<class T, class U>
void OutputParameter::setValue(int64_t idx, const U& rhs) {
    return dynamic_cast<TypedOutput<T>&>(*this).setValue(idx, rhs);
}

#endif  // DYNAMICLIBOP_H_
