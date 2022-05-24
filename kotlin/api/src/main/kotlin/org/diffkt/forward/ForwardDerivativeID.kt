/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.forward

import org.diffkt.DerivativeID
import org.diffkt.Shape

open class ForwardDerivativeID protected constructor() : DerivativeID() {
    constructor(inputTangentShapeForJacobian: Shape) : this() {
        // inputTangentShapeForJacobian should be empty / ignored to compute an output tangent (JVP),
        // it should only be used to compute a Jacobian
        this.inputTangentShapeForJacobian = inputTangentShapeForJacobian
    }

    private var savedInputTangentShapeForJacobian: Shape? = null
    var inputTangentShapeForJacobian: Shape
        get() = savedInputTangentShapeForJacobian!!
        protected set(value) { assert(savedInputTangentShapeForJacobian == null); savedInputTangentShapeForJacobian = value }
}