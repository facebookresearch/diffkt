/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/torch.h>
#include <iostream>

namespace F = torch::nn::functional;

void do_grads() {
    auto device = torch::kCUDA;
    auto options = torch::TensorOptions().device(device).requires_grad(true);

    // Add
    std::cout << std::endl;
    auto a = torch::ones({2, 3}, options);
    auto b = torch::ones({2, 3}, options);
    auto c = a + b;
    std::cout << c << std::endl;
    std::cout << c.grad_fn()->name() << std::endl;
    c.backward(torch::ones({2, 3}).to(device));
    std::cout << a.grad() << std::endl;

    // Sum (call .backward() on scalar)
    std::cout << std::endl;
    auto x = torch::ones({2, 3}, options);
    auto y = x.sum();
    std::cout << y << std::endl;
    std::cout << y.grad_fn()->name() << std::endl;
    y.backward();
    std::cout << x.grad() << std::endl;
}

torch::Tensor *heap_ones() {
    torch::Tensor ones = torch::ones({2, 2}).cuda();
    return new torch::Tensor(ones);
}

void heap_tensor_test() {
    torch::Tensor *v = heap_ones();
    std::cout << *v << std::endl;
    delete v;
}

torch::Tensor *heap_blob() {
    std::vector<float> vec = {-1, 0, 1, 2};
    auto t = torch::from_blob(vec.data(), {2, 2}).cuda().requires_grad_();
    return new torch::Tensor(t);
}

int main() {
    auto t = *heap_blob();
    auto u = t.relu();
    auto u_next = u.detach().requires_grad_();
    auto v = u_next.relu();
    auto seed = *heap_ones();
    v.backward(seed);
    std::cout << u_next.grad() << std::endl;
    u.backward(u_next.grad());
    std::cout << t.grad() << std::endl;
}
