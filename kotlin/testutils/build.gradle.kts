/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

val kotestVersion: String by System.getProperties()

dependencies {
    implementation("io.kotest:kotest-runner-junit5-jvm:$kotestVersion")
        ?.because("kotest framework")
    implementation(project(":api"))
}
