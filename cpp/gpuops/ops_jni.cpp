/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "ops_jni.h"

#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>

namespace F = torch::nn::functional;

// --- Helpers ---

// Makes a torch::Tensor (which is shared_ptr under the hood)
// stored on the heap, whose address can be cast to a long and
// passed to Java as a handle for the tensor.
//
// The handle must later be deleted via deleteHandle by Java.
at::Tensor *makeHandle(at::Tensor t) {
    return new torch::Tensor(t);
}

std::vector<int64_t> get_shape(JNIEnv *env, jintArray jshape) {
    int32_t rank = env->GetArrayLength(jshape);
    int32_t *dims = (jint *)env->GetPrimitiveArrayCritical(jshape, 0);
    std::vector<int64_t> shape{dims, dims + rank};
    env->ReleasePrimitiveArrayCritical(jshape, dims, 0);
    return shape;
}

std::vector<int64_t> to_long_vector(JNIEnv *env, jintArray jarr) {
    int32_t size = env->GetArrayLength(jarr);
    int32_t *data = (jint *)env->GetPrimitiveArrayCritical(jarr, 0);
    std::vector<int64_t> vec{data, data + size};
    env->ReleasePrimitiveArrayCritical(jarr, data, 0);
    return vec;
}

// --- Tensor utils ---

JNIEXPORT void JNICALL Java_org_diffkt_external_Gpu_deleteHandle(JNIEnv *, jobject, jlong handle) {
    delete (at::Tensor *)handle;
}

JNIEXPORT jintArray JNICALL Java_org_diffkt_external_Gpu_getShape(JNIEnv *env, jobject obj, jlong handle) {
    auto t = (at::Tensor *)handle;
    std::vector<int32_t> shape = {t->sizes().begin(), t->sizes().end()};
    jintArray jshape = env->NewIntArray(t->dim());
    env->SetIntArrayRegion(jshape, 0, t->dim(), shape.data());
    return jshape;
}

JNIEXPORT jfloatArray JNICALL Java_org_diffkt_external_Gpu_getFloatData(JNIEnv *env, jobject obj, jlong handle) {
    auto t = (at::Tensor *)handle;
    jfloatArray jdata = env->NewFloatArray(t->numel());
    env->SetFloatArrayRegion(jdata, 0, t->numel(), t->contiguous().cpu().data_ptr<float>());
    return jdata;
}

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_putFloatTensor(JNIEnv *env, jobject obj, jintArray jshape, jfloatArray jdata) {
    auto shape = get_shape(env, jshape);

    float *data = (jfloat *)env->GetPrimitiveArrayCritical(jdata, 0);
    auto t = torch::from_blob(data, shape).cuda();
    env->ReleasePrimitiveArrayCritical(jdata, data, 0);

    return (jlong)makeHandle(t);
}

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_zeros(JNIEnv *env, jobject obj, jintArray jshape) {
    auto shape = get_shape(env, jshape);
    auto t = torch::zeros(shape).cuda();
    return (jlong)makeHandle(t);
}

// --- Misc utils

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_getAllocatedBytes(JNIEnv *, jobject) {
    const c10::cuda::CUDACachingAllocator::StatType AGGREGATE = c10::cuda::CUDACachingAllocator::StatType::AGGREGATE;
    auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
    return stats.allocated_bytes[static_cast<size_t>(AGGREGATE)].current;
}

// --- Ops ---

JNIEXPORT jlongArray JNICALL Java_org_diffkt_external_Gpu_add(JNIEnv *env, jobject obj, jlong jlhs, jlong jrhs) {
    auto lhs = *(at::Tensor *)jlhs;
    auto rhs = *(at::Tensor *)jrhs;
    auto detached_lhs = lhs.detach().requires_grad_();
    auto detached_rhs = rhs.detach().requires_grad_();
    auto res = detached_lhs + detached_rhs;

    size_t returns = 3;
    jlong detached_and_res[returns];
    detached_and_res[0] = (jlong)makeHandle(detached_lhs);
    detached_and_res[1] = (jlong)makeHandle(detached_rhs);
    detached_and_res[2] = (jlong)makeHandle(res);

    jlongArray ret = env->NewLongArray(returns);
    env->SetLongArrayRegion(ret, 0, returns, detached_and_res);
    return ret;
}

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_addGradLhs(JNIEnv *env, jobject, jlong jseed, jlong jlhs, jlong jforward_res) {
    auto seed = (at::Tensor *)jseed;
    auto lhs = (at::Tensor *)jlhs;
    auto forward_res = (at::Tensor *)jforward_res;
    if (!lhs->grad().defined()) {
        forward_res->backward(*seed);
    }
    jlong ret = (jlong)makeHandle(lhs->grad());
    return ret;
}

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_addGradRhs(JNIEnv *env, jobject, jlong jseed, jlong jrhs, jlong jforward_res) {
    auto seed = (at::Tensor *)jseed;
    auto rhs = (at::Tensor *)jrhs;
    auto forward_res = (at::Tensor *)jforward_res;
    if (!rhs->grad().defined()) {
        forward_res->backward(*seed);
    }
    jlong ret = (jlong)makeHandle(rhs->grad());
    return ret;
}

JNIEXPORT jlongArray JNICALL Java_org_diffkt_external_Gpu_avgPool(JNIEnv *env, jobject, jlong jx, jint pool_height, jint pool_width) {
    auto x = *(at::Tensor *)jx;
    // Transpose to convert NHWC to NCHW (and detach and require grad as usual)
    auto nchw_input = x.detach().permute({0, 3, 1, 2}).requires_grad_();
    auto nchw_res = F::avg_pool2d(nchw_input, F::AvgPool2dFuncOptions({pool_height, pool_width}));
    auto nhwc_res = nchw_res.detach().permute({0, 2, 3, 1});

    // We return both nchw_res and nhwc_res; the former for passing to avgPoolGrad with
    // the detached and transformed input, and the latter for the actual op result in DiffKt
    size_t returns = 3;
    jlong input_and_res[returns];
    input_and_res[0] = (jlong)makeHandle(nchw_input);
    input_and_res[1] = (jlong)makeHandle(nchw_res);
    input_and_res[2] = (jlong)makeHandle(nhwc_res);

    jlongArray ret = env->NewLongArray(returns);
    env->SetLongArrayRegion(ret, 0, returns, input_and_res);
    return ret;
}

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_avgPoolGrad(JNIEnv *env, jobject, jlong jseed, jlong jforward_arg, jlong jforward_res) {
    auto seed = *(at::Tensor *)jseed;
    auto forward_arg = *(at::Tensor *)jforward_arg;
    auto forward_res = *(at::Tensor *)jforward_res;

    // Convert seed from NHWC to NCHW
    auto nchw_seed = seed.permute({0, 3, 1, 2});

    forward_res.backward(nchw_seed);
    auto nchw_grad = forward_arg.grad();

    // Convert grad from NCHW to NHWC
    auto nhwc_grad = nchw_grad.permute({0, 2, 3, 1});

    return (jlong)makeHandle(nhwc_grad);
}

JNIEXPORT jlongArray JNICALL Java_org_diffkt_external_Gpu_batchNorm2d(JNIEnv *env, jobject, jlong jinput, jlong jscale_shift, jlong jrunning_mean, jlong jrunning_variance, jfloat momentum) {
    auto input = *(at::Tensor *)jinput;
    auto scale_shift = (*(at::Tensor *)jscale_shift).detach().requires_grad_();
    auto running_mean = *(at::Tensor *)jrunning_mean;
    auto running_variance = *(at::Tensor *)jrunning_variance;

    // Convert NHWC to NCHW (and detach and require grad as usual)
    auto nchw_input = input.detach().permute({0, 3, 1, 2,}).requires_grad_();

    auto scale = scale_shift.slice(0, 0, 1);
    auto shift = scale_shift.slice(0, 1, 2);

    // Note: We use a momentum of 1 to get the same behavior as the CPU version. A momentum of 1
    // will replace the previous mean and variance with the current ones instead of behaving
    // like a running mean or variance. The default value for pytorch is 0.1
    auto options = F::BatchNormFuncOptions().weight(scale).bias(shift).momentum(momentum).training(true);
    auto nchw_res = F::batch_norm(nchw_input, running_mean, running_variance, options);
    auto res = nchw_res.detach().permute({0, 2, 3, 1});

    size_t returns = 4;
    jlong detached_and_res[returns];
    detached_and_res[0] = (jlong)makeHandle(nchw_input);
    detached_and_res[1] = (jlong)makeHandle(scale_shift);
    detached_and_res[2] = (jlong)makeHandle(nchw_res);
    detached_and_res[3] = (jlong)makeHandle(res);

    jlongArray ret = env->NewLongArray(returns);
    env->SetLongArrayRegion(ret, 0, returns, detached_and_res);
    return ret;
}

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_batchNorm2dGradInput(JNIEnv *env, jobject, jlong jseed, jlong jinput, jlong jforward_res) {
    auto seed = *(at::Tensor *)jseed;
    auto input = *(at::Tensor *)jinput;
    auto forward_res = *(at::Tensor *)jforward_res;

    // Convert seed from NHWC to NCHW
    auto nchw_seed = seed.permute({0, 3, 1, 2});

    if (!input.grad().defined())
        forward_res.backward(nchw_seed);
    auto nchw_grad = input.grad();

    // Convert grad from NCHW to NHWC
    auto nhwc_grad = nchw_grad.permute({0, 2, 3, 1});

    return (jlong)makeHandle(nhwc_grad);
}

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_batchNorm2dGradScaleShift(JNIEnv *env, jobject, jlong jseed, jlong jscale_shift, jlong jforward_res) {
    auto seed = *(at::Tensor *)jseed;
    auto scale_shift = *(at::Tensor *)jscale_shift;
    auto forward_res = *(at::Tensor *)jforward_res;

    // Convert seed from NHWC to NCHW
    auto nchw_seed = seed.permute({0, 3, 1, 2});

    if (!scale_shift.grad().defined()) {
        forward_res.backward(nchw_seed);
    }
    auto grad = scale_shift.grad();

    return (jlong)makeHandle(grad);
}

JNIEXPORT jlongArray JNICALL Java_org_diffkt_external_Gpu_broadcastTo(JNIEnv *env, jobject, jlong handle, jintArray jnew_shape) {
    auto t = (at::Tensor *)handle;
    auto detached = t->detach().requires_grad_();
    auto new_shape = to_long_vector(env, jnew_shape);
    // We use the expand as the implementation because broadcastTo was added in pytorch 1.8.0 which does not support cuda 9.2
    // TODO: When we get a machine with Cuda > 10.0 change to broadcast_to
    auto res = detached.expand(new_shape);

    jlong detached_and_res[2];
    detached_and_res[0] = (jlong)makeHandle(detached);
    detached_and_res[1] = (jlong)makeHandle(res);

    jlongArray ret = env->NewLongArray(2);
    env->SetLongArrayRegion(ret, 0, 2, detached_and_res);
    return ret;
}

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_broadcastToGrad(JNIEnv *env, jobject, jlong jseed, jlong jforward_arg, jlong jforward_res) {
    auto seed = (at::Tensor *)jseed;
    auto forward_arg = (at::Tensor *)jforward_arg;
    auto forward_res = (at::Tensor *)jforward_res;
    forward_res->backward(*seed);
    jlong ret = (jlong)makeHandle(forward_arg->grad());
    return ret;
}

JNIEXPORT jlongArray JNICALL Java_org_diffkt_external_Gpu_conv2d(JNIEnv *env, jobject, jlong jimages, jlong jfilters, jintArray jstrides, jintArray jpadding) {
    auto images = *(at::Tensor *)jimages;
    auto filters = *(at::Tensor *)jfilters;
    auto strides = to_long_vector(env, jstrides);
    auto padding = to_long_vector(env, jpadding);

    // Convert NHWC to NCHW (and detach and require grad as usual)
    auto nchw_images = images.detach().permute({0, 3, 1, 2}).requires_grad_();
    // Convert OHWI to OIHW
    auto oihw_filters = filters.detach().permute({0, 3, 1, 2}).requires_grad_();
    // Rearrange padding. We get (top, bottom, left, right), but we want
    // (left, right, top, bottom).
    assert(padding.size() == 4);
    std::vector<int64_t> ordered_padding = {padding[2], padding[3], padding[0], padding[1]};

    // Pad the image before we do conv2d because conv2d padding only supports
    // (height, width() padding, not (top, bottom, left, right).
    auto padded_images = F::pad(nchw_images, F::PadFuncOptions(ordered_padding));
    auto nchw_res = F::conv2d(padded_images, oihw_filters, F::Conv2dFuncOptions().stride(strides));
    auto res = nchw_res.detach().permute({0, 2, 3, 1});

    size_t returns = 4;
    jlong detached_and_res[returns];
    detached_and_res[0] = (jlong)makeHandle(nchw_images);
    detached_and_res[1] = (jlong)makeHandle(oihw_filters);
    detached_and_res[2] = (jlong)makeHandle(nchw_res);
    detached_and_res[3] = (jlong)makeHandle(res);

    jlongArray ret = env->NewLongArray(returns);
    env->SetLongArrayRegion(ret, 0, returns, detached_and_res);
    return ret;
}

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_conv2dGradImages(JNIEnv *env, jobject, jlong jseed, jlong jimages, jlong jforward_res) {
    auto seed = (at::Tensor *)jseed;
    auto images = (at::Tensor *)jimages;
    auto forward_res = (at::Tensor *)jforward_res;

    // Convert seed from NHWC to NCHW
    auto nchw_seed = seed->permute({0, 3, 1, 2});

    if (!images->grad().defined()) {
        forward_res->backward(nchw_seed);
    }
    auto nchw_grad = images->grad();

    // Convert grad from NCHW to NHWC
    auto nhwc_grad = nchw_grad.permute({0, 2, 3, 1});

    return (jlong)makeHandle(nhwc_grad);
}

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_conv2dGradFilters(JNIEnv *env, jobject, jlong jseed, jlong jfilters, jlong jforward_res) {
    auto seed = (at::Tensor *)jseed;
    auto filters = (at::Tensor *)jfilters;
    auto forward_res = (at::Tensor *)jforward_res;

    // Convert seed from NHWC to NCHW
    auto nchw_seed = seed->permute({0, 3, 1, 2});

    if (!filters->grad().defined()) {
        forward_res->backward(nchw_seed);
    }
    auto nchw_grad = filters->grad();

    // Convert grad from NCHW to NHWC
    auto nhwc_grad = nchw_grad.permute({0, 2, 3, 1});

    return (jlong)makeHandle(nhwc_grad);
}

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_div(JNIEnv *env, jobject, jlong jlhs, jlong jrhs) {
    auto lhs = *(at::Tensor *)jlhs;
    auto rhs = *(at::Tensor *)jrhs;
    auto res = lhs.detach() / rhs.detach();
    return (jlong)makeHandle(res);
}

JNIEXPORT jlongArray JNICALL Java_org_diffkt_external_Gpu_logSoftmax(JNIEnv *env, jobject, jlong jx, jint axis) {
    auto x = *(at::Tensor *)jx;
    auto detached_x = x.detach().requires_grad_();
    auto res = torch::log_softmax(detached_x, axis);

    size_t returns = 2;
    jlong detached_and_res[returns];
    detached_and_res[0] = (jlong)makeHandle(detached_x);
    detached_and_res[1] = (jlong)makeHandle(res);

    jlongArray ret = env->NewLongArray(returns);
    env->SetLongArrayRegion(ret, 0, returns, detached_and_res);
    return ret;
}

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_logSoftmaxGrad(JNIEnv *env, jobject, jlong jseed, jlong jx, jlong jforward_res) {
    auto seed = (at::Tensor *)jseed;
    auto x = (at::Tensor *)jx;
    auto forward_res = (at::Tensor *)jforward_res;
    forward_res->backward(*seed);
    auto ret = (jlong)makeHandle(x->grad());
    return ret;
}

JNIEXPORT jlongArray JNICALL Java_org_diffkt_external_Gpu_matmul(JNIEnv *env, jobject, jlong jlhs, jlong jrhs) {
    auto lhs = *(at::Tensor *)jlhs;
    auto rhs = *(at::Tensor *)jrhs;
    auto detached_lhs = lhs.detach().requires_grad_();
    auto detached_rhs = rhs.detach().requires_grad_();
    auto res = detached_lhs.matmul(detached_rhs);

    size_t returns = 3;
    jlong detached_and_res[returns];
    detached_and_res[0] = (jlong)makeHandle(detached_lhs);
    detached_and_res[1] = (jlong)makeHandle(detached_rhs);
    detached_and_res[2] = (jlong)makeHandle(res);

    jlongArray ret = env->NewLongArray(returns);
    env->SetLongArrayRegion(ret, 0, returns, detached_and_res);
    return ret;
}

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_matmulGradLhs(JNIEnv *env, jobject, jlong jseed, jlong jlhs, jlong jforward_res) {
    auto seed = (at::Tensor *)jseed;
    auto lhs = (at::Tensor *)jlhs;
    auto forward_res = (at::Tensor *)jforward_res;
    if (!lhs->grad().defined()) {
        forward_res->backward(*seed);
    }
    jlong ret = (jlong)makeHandle(lhs->grad());
    return ret;
}

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_matmulGradRhs(JNIEnv *env, jobject, jlong jseed, jlong jrhs, jlong jforward_res) {
    auto seed = (at::Tensor *)jseed;
    auto rhs = (at::Tensor *)jrhs;
    auto forward_res = (at::Tensor *)jforward_res;
    if (!rhs->grad().defined()) {
        forward_res->backward(*seed);
    }
    jlong ret = (jlong)makeHandle(rhs->grad());
    return ret;
}

JNIEXPORT jlongArray JNICALL Java_org_diffkt_external_Gpu_maxPool(JNIEnv *env, jobject, jlong jx, jint pool_height, jint pool_width) {
    auto x = *(at::Tensor *)jx;
    // Transpose to convert NHWC to NCHW (and detach and require grad as usual)
    auto nchw_input = x.detach().permute({0, 3, 1, 2}).requires_grad_();
    auto nchw_res = F::max_pool2d(nchw_input, F::MaxPool2dFuncOptions({pool_height, pool_width}));
    auto nhwc_res = nchw_res.detach().permute({0, 2, 3, 1});

    // We return both nchw_res and nhwc_res; the former for passing to maxPoolGrad with
    // the detached and transformed input, and the latter for the actual op result in DiffKt
    size_t returns = 3;
    jlong input_and_res[returns];
    input_and_res[0] = (jlong)makeHandle(nchw_input);
    input_and_res[1] = (jlong)makeHandle(nchw_res);
    input_and_res[2] = (jlong)makeHandle(nhwc_res);

    jlongArray ret = env->NewLongArray(returns);
    env->SetLongArrayRegion(ret, 0, returns, input_and_res);
    return ret;
}

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_maxPoolGrad(JNIEnv *env, jobject, jlong jseed, jlong jforward_arg, jlong jforward_res) {
    auto seed = (at::Tensor *)jseed;
    auto forward_arg = (at::Tensor *)jforward_arg;
    auto forward_res = (at::Tensor *)jforward_res;

    // Convert seed from NHWC to NCHW
    auto nchw_seed = seed->permute({0, 3, 1, 2});

    forward_res->backward(nchw_seed);
    auto nchw_grad = forward_arg->grad();

    // Convert grad from NCHW to NHWC
    auto nhwc_grad = nchw_grad.permute({0, 2, 3, 1});

    return (jlong)makeHandle(nhwc_grad);
}

JNIEXPORT jlongArray JNICALL Java_org_diffkt_external_Gpu_nllLoss(JNIEnv *env, jobject, jlong jx, jlong jlabels) {
    auto x = *(at::Tensor *)jx;
    auto labels = *(at::Tensor *)jlabels;
    auto detached_x = x.detach().requires_grad_();
    auto detached_labels = labels.detach().requires_grad_();
    auto res = torch::nll_loss(detached_x, detached_labels.toType(torch::kLong));

    size_t returns = 3;
    jlong detached_and_res[returns];
    detached_and_res[0] = (jlong)makeHandle(detached_x);
    detached_and_res[1] = (jlong)makeHandle(detached_labels);
    detached_and_res[2] = (jlong)makeHandle(res);

    jlongArray ret = env->NewLongArray(returns);
    env->SetLongArrayRegion(ret, 0, returns, detached_and_res);
    return ret;
}

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_nllLossGradX(JNIEnv *env, jobject, jlong jseed, jlong jx, jlong jforward_res) {
    auto seed = (at::Tensor *)jseed;
    auto x = (at::Tensor *)jx;
    auto forward_res = (at::Tensor *)jforward_res;
    if (!x->grad().defined()) {
        forward_res->backward(*seed);
    }
    auto ret = (jlong)makeHandle(x->grad());
    return ret;
}

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_nllLossGradLabels(JNIEnv *env, jobject, jlong jseed, jlong jlabels, jlong jforward_res) {
    auto seed = (at::Tensor *)jseed;
    auto labels = (at::Tensor *)jlabels;
    auto forward_res = (at::Tensor *)jforward_res;
    if (!labels->grad().defined()) {
        forward_res->backward(*seed);
    }
    auto ret = (jlong)makeHandle(labels->grad());
    return ret;
}

JNIEXPORT jlongArray JNICALL Java_org_diffkt_external_Gpu_relu(JNIEnv *env, jobject obj, jlong handle) {
    auto t = (at::Tensor *)handle;
    auto detached = t->detach().requires_grad_();
    auto res = detached.relu();

    jlong detached_and_res[2];
    detached_and_res[0] = (jlong)makeHandle(detached);
    detached_and_res[1] = (jlong)makeHandle(res);

    jlongArray ret = env->NewLongArray(2);
    env->SetLongArrayRegion(ret, 0, 2, detached_and_res);
    return ret;
}

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_reluGrad(JNIEnv *env, jobject obj, jlong jseed, jlong jforward_arg, jlong jforward_result) {
    auto seed = (at::Tensor *)jseed;
    auto forward_arg = (at::Tensor *)jforward_arg;
    auto forward_result = (at::Tensor *)jforward_result;
    forward_result->backward(*seed);
    auto heap_grad = new torch::Tensor(forward_arg->grad());
    return (jlong)heap_grad;
}

JNIEXPORT jlongArray JNICALL Java_org_diffkt_external_Gpu_reshape(JNIEnv *env, jobject, jlong handle, jintArray jnew_shape) {
    auto t = (at::Tensor *)handle;
    auto detached = t->detach().requires_grad_();
    auto new_shape = to_long_vector(env, jnew_shape);
    auto res = detached.reshape(new_shape);

    jlong detached_and_res[2];
    detached_and_res[0] = (jlong)makeHandle(detached);
    detached_and_res[1] = (jlong)makeHandle(res);

    jlongArray ret = env->NewLongArray(2);
    env->SetLongArrayRegion(ret, 0, 2, detached_and_res);
    return ret;
}

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_reshapeGrad(JNIEnv *env, jobject, jlong jseed, jlong jforward_arg, jlong jforward_res) {
    auto seed = (at::Tensor *)jseed;
    auto forward_arg = (at::Tensor *)jforward_arg;
    auto forward_res = (at::Tensor *)jforward_res;
    forward_res->backward(*seed);
    jlong ret = (jlong)makeHandle(forward_arg->grad());
    return ret;
}

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_sub(JNIEnv *env, jobject, jlong jlhs, jlong jrhs) {
    auto lhs = *(at::Tensor *)jlhs;
    auto rhs = *(at::Tensor *)jrhs;
    auto res = lhs.detach() - rhs.detach();
    return (jlong)makeHandle(res);
}

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_isub(JNIEnv *env, jobject, jlong jlhs, jlong jrhs) {
    auto lhs = *(at::Tensor *)jlhs;
    auto rhs = *(at::Tensor *)jrhs;
    lhs.sub_(rhs);
    return jlhs;
}

JNIEXPORT jlongArray JNICALL Java_org_diffkt_external_Gpu_sum(JNIEnv *env, jobject, jlong handle, jintArray jaxes, jboolean jkeep_dims) {
    auto t = (at::Tensor *)handle;
    auto detached = t->detach().requires_grad_();
    auto axes = to_long_vector(env, jaxes);
    auto keep_dims = (bool) jkeep_dims;
    auto res = detached.sum(axes, keep_dims);

    jlong detached_and_res[2];
    detached_and_res[0] = (jlong)makeHandle(detached);
    detached_and_res[1] = (jlong)makeHandle(res);

    jlongArray ret = env->NewLongArray(2);
    env->SetLongArrayRegion(ret, 0, 2, detached_and_res);
    return ret;
}
JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_sumGrad(JNIEnv *, jobject, jlong jseed, jlong jforward_arg, jlong jforward_res) {
    auto seed = (at::Tensor *)jseed;
    auto forward_arg = (at::Tensor *)jforward_arg;
    auto forward_res = (at::Tensor *)jforward_res;
    forward_res->backward(*seed);
    jlong ret = (jlong)makeHandle(forward_arg->grad());
    return ret;
}

JNIEXPORT jlong JNICALL Java_org_diffkt_external_Gpu_times(JNIEnv *env, jobject, jlong jlhs, jlong jrhs) {
    auto lhs = *(at::Tensor *)jlhs;
    auto rhs = *(at::Tensor *)jrhs;
    auto res = lhs.detach() * rhs.detach();
    return (jlong)makeHandle(res);
}
