/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

/**
 * Implement [Differentiable] to have the runtime be capable of using your data type
 * as an input or output type of a function that can be differentiated.
 */
interface Differentiable<out T: Differentiable<T>>: Wrappable<T>
