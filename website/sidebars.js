/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

module.exports = {

  diffktSidebar: [
    {
      type: 'category',
      label: 'Overview',
      collapsible: true,
      collapsed: true,
      items: [
        {
          type: 'doc',
          id: 'overview/why_diffkt/why_diffkt',
          label: 'Why Diffkt?',
        },
        {
          type: 'doc',
          id: 'overview/automatic_differentiation/automatic_differentiation',
          label: 'Automatic Differentiation',
        },
        {
          type: 'doc',
          id: 'overview/quick_start/quick_start',
          label: 'Quick Start',
        },
        {
          type: 'doc',
          id: 'overview/installation/installation',
          label: 'Installation',
        },
      ],
    },
    {
      type: 'category',
      label: 'Framework',
      collapsible: true,
      collapsed: true,
      items: [
        {
          type: 'doc',
          id: 'framework/api/api',
          label: 'API',
        },
        {
          type: 'doc',
          id: 'framework/data_science/data_science',
          label: 'Data Science',
        },
        {
          type: 'doc',
          id: 'framework/physical_systems/physical_systems',
          label: 'Physical Systems',
        },
      ],
    },
    {
      type: 'category',
      label: 'Tutorials',
      collapsible: true,
      collapsed: true,
      items: [
        {
          type: 'category',
          label: 'API',
          collapsible: true,
          collapsed: true,
          items: [
            {
              type: 'doc',
              id: 'tutorials/api/intro_to_diff_prog/intro_to_diff_prog',
              label: 'Introduction',
            },
            {
              type: 'doc',
              id: 'tutorials/api/indexing/indexing',
              label: 'Indexing, Views, and Accessing Values',
            },
            {
              type: 'doc',
              id: 'tutorials/api/broadcasting/broadcasting',
              label: 'Broadcasting',
            },
            {
              type: 'doc',
              id: 'tutorials/api/user_defined_types/user_defined_types',
              label: 'User Defined Types',
            },
          ],
        },
        {
          type: 'category',
          label: 'Data Science',
          collapsible: true,
          collapsed: true,
          items: [
            {
              type: 'doc',
              id: 'tutorials/data_science/simple_parabola/simple_parabola',
              label: 'Gradient Descent on a Simple Parabola',
            },
            {
              type: 'doc',
              id: 'tutorials/data_science/mle/mle',
              label: 'Normal Distribution and Gradient Descent',
            },
            {
              type: 'doc',
              id: 'tutorials/data_science/linear_regression/linear_regression',
              label: 'Gradient Descent and Linear Regression',
            },
            {
              type: 'doc',
              id: 'tutorials/data_science/linear_regression_udt/linear_regression_udt',
              label: 'Linear Regression with User Defined Types',
            },
            {
              type: 'doc',
              id: 'tutorials/data_science/logistics_regression/logistics_regression',
              label: 'Logistics Regression and Gradient Descent',
            },
            {
              type: 'doc',
              id: 'tutorials/data_science/exponential_distribution/exponential_distribution',
              label: 'Exponential Distribution and Gradient Descent',
            },
            {
              type: 'doc',
              id: 'tutorials/data_science/multivariable_regression/multivariable_regression',
              label: 'Gradient Descent and Multivariable Linear Regression',
            },
            {
              type: 'doc',
              id: 'tutorials/data_science/nonlinear_regression/nonlinear_regression',
              label: '3D Nonlinear Regression',
            },
            {
              type: 'doc',
              id: 'tutorials/data_science/neural_network/neural_network',
              label: 'Neural Networks and Stochastic Gradient Descent',
            },
            {
              type: 'doc',
              id: 'tutorials/data_science/neural_network_udt/neural_network_udt',
              label: 'Neural Network User-Defined Types',
            },
          ],
        },
        {
          type: 'category',
          label: 'Physical Systems',
          collapsible: true,
          collapsed: true,
          items: [
            {
              type: 'doc',
              id: 'tutorials/physical_systems/mass_spring/mass_spring',
              label: 'Mass Spring System',
            },
            {
              type: 'doc',
              id: 'tutorials/physical_systems/mass_spring_jit/mass_spring_jit',
              label: 'Mass Spring System using Just In Time Optimization',
            },
            {
              type: 'doc',
              id: 'tutorials/physical_systems/soft_sphere/soft_sphere',
              label: 'Soft Sphere',
            },
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Examples',
      collapsible: true,
      collapsed: true,
      items: [
        {
          type: 'doc',
          id: 'examples/brachistochrone/brachistochrone',
          label: 'Brachistochrone Curve',
        },
        {
          type: 'doc',
          id: 'examples/conjugate_gradient/conjugate_gradient',
          label: 'Conjugate Gradient',
        },
        {
          type: 'doc',
          id: 'examples/custom_reverse/custom_reverse',
          label: 'Custom Reverse Automatic Differentiation Function',
        },
        {
          type: 'doc',
          id: 'examples/free_fall/free_fall',
          label: 'Free Falling Mass',
        },
        {
          type: 'doc',
          id: 'examples/french_to_english/french_to_english',
          label: 'French To English Translation',
        },
        {
          type: 'doc',
          id: 'examples/hookean_spring/hookean_spring',
          label: 'Hookean Spring',
        },
        {
          type: 'doc',
          id: 'examples/iris/iris',
          label: 'Iris',
        },
        {
          type: 'doc',
          id: 'examples/linear_regression/linear_regression',
          label: 'Linear Regression',
        },
        {
          type: 'doc',
          id: 'examples/mnist/mnist',
          label: 'MNIST',
        },
        {
          type: 'doc',
          id: 'examples/neohookean/neohookean',
          label: 'Neohookean',
        },
        {
          type: 'doc',
          id: 'examples/poisson_blending/poisson_blending',
          label: 'Poisson Blending',
        },
        {
          type: 'doc',
          id: 'examples/quadratic/quadratic',
          label: 'Quadratic',
        },
        {
          type: 'doc',
          id: 'examples/resnet/resnet',
          label: 'Resnet',
        },
        {
          type: 'doc',
          id: 'examples/root_two/root_two',
          label: 'Root Two',
        },
        {
          type: 'doc',
          id: 'examples/soft_sphere/soft_sphere',
          label: 'Soft Sphere',
        },
        {
          type: 'doc',
          id: 'examples/spring_area/spring_area',
          label: 'Spring Area',
        },
        {
          type: 'doc',
          id: 'examples/symbolic_diff/symbolic_diff',
          label: 'Symbolic Differentiation',
        },
        {
          type: 'doc',
          id: 'examples/vector2/vector2',
          label: 'Vector2',
        },
      ],
    },
    {
      type: 'category',
      label: 'Contribute',
      collapsible: true,
      collapsed: true,
      items: [
        {
          type: 'doc',
          id: 'contribute/contribute',
          label: 'Contribute to DiffKt',
        },
      ],
    },
  ],
};
