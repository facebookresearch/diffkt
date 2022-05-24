/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

/*
 * Support for user-defined types as the input or output of a derivative operation.
 */

/**
 * Implement [Wrappable] to have the runtime be capable of using your data type
 * as an input or output type of a function that can be wrapped.
 */
interface Wrappable<out T: Wrappable<T>> {
    /**
     * The wrap function should return the same static type it is declared on.
     */
    fun wrap(wrapper: Wrapper): T
}
