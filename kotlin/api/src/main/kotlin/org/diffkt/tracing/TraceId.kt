/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.tracing

/**
 * A [TraceId] is used to distinguish one trace from another.  That way the variables from one trace will not be
 * confused with the variables from another possibly nested trace, as they may share identifiers.
 */
class TraceId
