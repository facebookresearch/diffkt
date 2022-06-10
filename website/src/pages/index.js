/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './styles.module.css';

const features = [
  {
    title: 'Automatic Differentiation with Kotlin',
    description: (
      <>
        <p>
          DiffKt is a general-purpose, functional, differentiable programming framework for Kotlin.
          It can automatically differentiate through functions of tensors, scalars, and user-defined types.
          It supports forward-mode and reverse-mode differentiation including
          Jacobian-vector and vector-Jacobian products, which can be composed for higher-order differentiation.
        </p>
        <p>
          DiffKt also includes an early integration of ShapeTyping, an extensible
          compiler plugin for ahead-of-time tensor shape verification and inspection.
          With the ShapeTyping IntelliJ IDE plugin, users can even inspect tensor shapes
          and see tensor shape errors while in active development.
        </p>
      </>
    ),
  },
  {
    title: 'Functions over Tensors',
    description: (
      <>
        Multi-dimensional tensor data types have become popular with deep neural networks. They are used
        in many applications of data science, such as graph analysis and multi-way statistics.
        DiffKt support automatic differentiation of functions over tensors.
      </>
    ),
  },
  {
    title: 'Functions over User Defined Types',
    description: (
      <>
        DiffKt lets you create your own user defined types and complex data structures for functions,
        and can provide automatic differentiation of the functions. This is useful for complex simulations
        or physical systems modeling, where there are objects with multiple variables that can represent different
        entities in your simulation.
      </>
    ),
  },
];

function Feature({imageUrl, title, description}) {
  const imgUrl = useBaseUrl(imageUrl);
  return (
    <div className={clsx('col col--4', styles.feature)}>
      {imgUrl && (
        <div className="text--center">
          <img className={styles.featureImage} src={imgUrl} alt={title} />
        </div>
      )}
      <h3>{title}</h3>
      <p>{description}</p>
    </div>
  );
}

export default function Home() {
  const context = useDocusaurusContext();
  const {siteConfig = {}} = context;
  return (
    <Layout
      title={` ${siteConfig.title}`}
      description="A Kotlin Library for Automatic Differentiation <head />">
      <header className={clsx('hero hero--primary', styles.heroBanner)}>
        <div className="container">
          <h1 className="hero__title">{siteConfig.title}</h1>
          <p className="hero__subtitle">{siteConfig.tagline}</p>
          <div className={styles.buttons}>
            <Link
              className={clsx(
                'button button--outline button--secondary button--lg',
                styles.getStarted,
              )}
              to={useBaseUrl('docs/overview/why_diffkt')}>
              Get Started
            </Link>
          </div>
        </div>
      </header>
      <main>
        {features && features.length > 0 && (
          <section className={styles.features}>
            <div className="container">
              <div className="row">
                {features.map(({title, imageUrl, description}) => (
                  <Feature
                    key={title}
                    title={title}
                    imageUrl={imageUrl}
                    description={description}
                  />
                ))}
              </div>
            </div>
          </section>
        )}
      </main>
    </Layout>
  );
}
