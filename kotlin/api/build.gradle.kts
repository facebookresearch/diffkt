/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import org.jetbrains.dokka.gradle.DokkaTask

plugins {
    `maven-publish`
    val shapeTypingVersion = "0.1.0-3dc727b"
    id("org.diffkt.shapeTyping") version shapeTypingVersion
    id("org.jetbrains.dokka") version "1.6.0"
}

repositories {
    jcenter() // or maven(url="https://dl.bintray.com/kotlin/dokka")
    mavenCentral()
    maven {
        name = "GitHubPackages"
        url = uri("https://maven.pkg.github.com/facebookresearch/diffkt")
        credentials {
            username = System.getenv("GITHUB_ACTOR")
            password = System.getenv("GITHUB_TOKEN")
        }
    }
    mavenLocal()
    mavenCentral()
}

dependencies {
    implementation(group = "net.bytebuddy", name = "byte-buddy", version="1.12.7")
    compileOnly("org.diffkt.shapeTyping:annotations:0.1.0-3dc727b")
    compileOnly("org.diffkt.shapeTyping:shape-functions:0.1.0-3dc727b")
    testCompileOnly("org.diffkt.shapeTyping:annotations:0.1.0-3dc727b")
    testCompileOnly("org.diffkt.shapeTyping:shape-functions:0.1.0-3dc727b")
    testImplementation(project(":testutils"))
}

tasks.register<Exec>("buildExternalJni") {
    commandLine("../../cpp/ops/scripts/build.sh")
}

tasks.register<Exec>("buildGpuJni") {
    commandLine("../../cpp/gpuops/build.sh")
}

tasks.named("compileKotlin") {
    dependsOn("buildExternalJni")
    if (project.hasProperty("gpu")) {
        dependsOn("buildGpuJni")
    }
}

tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile>() {
    kotlinOptions.freeCompilerArgs += "-Xserialize-ir=inline"
}

// To publish locally, run the following commands:
//          export GITHUB_USERNAME=${{YOUR USRNAME}}
//          export GITHUB_ACCESS_TOKEN=${{ACCESS TOKEN WITH PACKAGE WRITE ACCESS}}
//          ./gradlew publish -Pversion=0.1.0-$(git rev-parse --short HEAD)
publishing {
    publications {
        create<MavenPublication>("maven") {
            groupId = "org.diffkt"
            artifactId = "api"
            version = project.version.toString()
            from(components["java"])
        }
    }
    repositories {
        maven {
            name = "GitHubPackages"
            url = uri("https://maven.pkg.github.com/facebookresearch/diffkt")
            credentials {
                username = System.getenv("GITHUB_ACTOR")
                password = System.getenv("GITHUB_TOKEN")
            }
        }
    }
}

apply(plugin="org.jetbrains.dokka")
tasks.withType<DokkaTask>().configureEach{
    outputDirectory.set(rootDir.resolve("../docs/api"))
    dokkaSourceSets {
      configureEach {
          samples.from(rootDir.resolve("samples/src/main/kotlin"))
      }
    }
}
