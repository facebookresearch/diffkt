/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import org.jetbrains.dokka.gradle.DokkaTask


plugins {
    `maven-publish`
    id("shapeKt") version "1.0"
    id("org.jetbrains.dokka") version "1.6.0"
    id("maven-publish")

    signing
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
    compileOnly("shapeKt:annotations:1.0")
    compileOnly("shapeKt:shape-functions:1.0")
    testCompileOnly("shapeKt:annotations:1.0")
    testCompileOnly("shapeKt:shape-functions:1.0")
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

/*
publishing {
    publications {
        create<MavenPublication>("maven") {
            groupId = "com.facebook"
            artifactId = "api"
            version = "0.0.1-DEV"//project.version.toString()
            from(components["java"])
        }
    }
    repositories {
        /*maven {
            name = "GitHubPackages"
            url = uri("https://maven.pkg.github.com/facebookresearch/diffkt")
            credentials {
                username = System.getenv("GITHUB_ACTOR")
                password = System.getenv("GITHUB_TOKEN")
            }
        }*/
        if (version.toString().endsWith("SNAPSHOT")) {
            maven("https://s01.oss.sonatype.org/content/repositories/snapshots/") {
                name = "sonatypeReleaseRepository"
                credentials {
                    username = properties.get("repositoryUsername") as String
                    password = properties.get("repositoryPassword") as String
                }
            }
        } else {
            maven("https://s01.oss.sonatype.org/service/local/staging/deploy/maven2/") {
                name = "sonatypeSnapshotRepository"
                credentials {
                    username = properties.get("repositoryUsername") as String
                    password = properties.get("repositoryPassword") as String
                }
            }
        }
    }
}
*/
val releasesRepoUrl = "https://s01.oss.sonatype.org/service/local/staging/deploy/maven2/"
val snapshotsRepoUrl = "https://s01.oss.sonatype.org/content/repositories/snapshots/"


fun getRepoUrl(): String {
    if (version.toString().endsWith("SNAPSHOT")) {
        return snapshotsRepoUrl
    }
    return releasesRepoUrl
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            groupId = "com.facebook"
            artifactId = "api"
            version = "0.0.1-DEV1"//project.version.toString()
            from(components["java"])
        }
    }

    repositories {
        /*withType<MavenArtifactRepository> {
            if (name == "local") {
                return@withType
            }
            url = uri(getRepoUrl())
            credentials {
                username = project.property("repositoryUsername").toString()
                password = project.property("repositoryPassword").toString()
            }
        }*/
        maven(getRepoUrl()) {
            name = "sonatypeReleaseRepository"
            credentials {
                username = project.property("repositoryUsername").toString()
                password = project.property("repositoryPassword").toString()
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
