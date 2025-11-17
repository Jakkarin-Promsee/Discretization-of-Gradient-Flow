# Hessian Optimization Theory in Deep Learning

Prepared by

# Hessian Optimization Theory in Deep Learning

Prepared by

Mr. Jakkarin Promsri 67070501009  
Mr. Teekatas Wongsuesantati 67070501019  
Mr. Thawat Boonsuk 67070501024  
Mr. Teerakarn Noiraksa 67070501062  
Mr. Achiravit Prasom 67070501076  
Mr. Pongsatorn Puttasorn 67070501084

Submitted to

Dr. Anuwat Tangthanawatsakul

A report for the course **MTH 234 Linear Algebra**  
King Mongkut’s University of Technology Thonburi  
Semester 1, Academic Year 2025

---

## Abstract

The Hessian matrix **H** is a square symmetric matrix of second-order partial derivatives of a multivariable scalar function \(f(x)\). It serves as a fundamental tool for analyzing local curvature in multidimensional spaces. The matrix is used as the core component for approximating a function around a critical point using a quadratic form through the Taylor expansion.

The mathematical characteristics of H are crucial due to the following reasons:

1. **Symmetry** ensures that eigenvalues are real and allows stable inversion of the matrix \(H^{-1}\).
2. **Eigenvectors** represent principal curvature directions, and **eigenvalues** indicate curvature magnitude, enabling the construction of an _orthonormal basis_ for deeper analysis.
3. The matrix–vector product \(Hv\) directly measures the effect of curvature along direction \(v\).

These components assist in identifying critical points (local minima, maxima, saddle points) and form the foundation of second-order algorithms such as Newton’s Method and Quasi-Newton Methods.

---

## Introduction

Linear Algebra is a core foundation of modern computation, playing a vital role in multidimensional data processing, signal computation, large-scale data analytics, and algorithm design. Deep Learning in particular heavily relies on vector and matrix structures and linear transformations.

One prominent topic is the analysis of the curvature of a loss function through the _Hessian Matrix_. This matrix captures the curvature structure of the loss landscape, enabling identification of minima, saddle points, and maxima, and serves as the basis for second-order optimization.

This report explains the role of Linear Algebra in Deep Learning and the significance of the Hessian in analyzing and improving training performance.

---

## Mathematical Concepts from Linear Algebra

- **Matrix-Vector Product**: Used to evaluate the effect of \(H\) on a vector \(v\) without explicitly forming H
- **Orthogonal / Orthonormal Basis**: Essential for eigen decomposition
- **Eigenvalues / Eigenvectors**: Describe directions and magnitudes of curvature
- **Eigen Basis**: A basis constructed from eigenvectors
- **Hessian Matrix (H)**: A symmetric matrix of second-order derivatives
- **Linear Transformation**: Viewing the Hessian as a linear operator
- **Taylor Series**: Used to approximate the loss with a quadratic form

---

## Detail Description

### 1. Principles of Deep Learning

#### 1.1 Neural Networks and Non-linear Relations

Explanation of layer structure, activation functions, and non-linear relationships.

#### 1.2 Forward Pass

Basic equations:

\[
z^{(l)} = a^{(l-1)}W^{(l)} + b^{(l)}
\]
\[
a^{(l)} = f(z^{(l)})
\]

#### Loss Computation

Using Mean Square Error:

\[
L = \frac{1}{N} \sum_i (y_i - \hat{y}\_i)^2
\]

---

### 2. Gradient via First-Order Derivatives

Parameter updates using gradient descent:

\[
W \leftarrow W - \eta \nabla_W L
\]

Using the chain rule to compute the backward pass (backpropagation).

---

### 3. Role of the Hessian in Deep Learning

#### 3.1 Definition and Dimension of the Hessian

\(
\nabla L(\theta) \in \mathbb{R}^p
\)  
\(
H = \nabla^2 L(\theta) \in \mathbb{R}^{p \times p}
\)

#### 3.2 Key Properties of the Hessian

- **Clairaut's Theorem**: mixed partial derivatives are equal → H is symmetric
- **Spectral Theorem**:

\[
H = Q \Lambda Q^T
\]

where \(Q\) is orthogonal and \(\Lambda\) is diagonal.

#### 3.3 Using the Hessian in Taylor Series

\[
L(\theta + \Delta) \approx
L(\theta) + \nabla L^T \Delta + \frac{1}{2}\Delta^T H \Delta
\]

Used to analyze curvature and determine optimal update directions.

---

## Application Example: Deep Autoencoder

An example using Hessian-Free Optimization (HF).

### Main Steps

1. Form the linear system:
   \[
   H p = -\nabla L
   \]
2. Use Conjugate Gradient (CG) to solve it without forming H
3. Use Hessian-vector products \(Hv\)
4. Update parameters using the direction \(p\)

Result: Converges faster than SGD, especially in ill-conditioned regions.

---

## Discussion and Analysis

### Advantages

- Effective in areas with complex curvature
- Requires fewer epochs

### Limitations

- H may not be positive-definite
- Higher computational cost than first-order methods
- CG requires tuning

---

## Conclusion

The Hessian matrix provides insight into the curvature of the loss landscape and enhances training efficiency through second-order methods. Although models can be large, techniques such as Hessian-vector products, FIM, and K-FAC make second-order optimization feasible in practice.

---

## References

- Frans, K. (2023). _Second-Order Optimization_. Notes on Deep Learning.
- KMUTT. (n.d.). _MTH 234 Course Material_.
- Martens, J. (2010). _Deep learning via Hessian-free optimization_.

Mr. Jakkarin Promsri 67070501009  
Mr. Teekatas Wongsuesantati 67070501019  
Mr. Thawat Boonsuk 67070501024  
Mr. Teerakarn Noiraksa 67070501062  
Mr. Achiravit Prasom 67070501076  
Mr. Pongsatorn Puttasorn 67070501084

Submitted to

Dr. Anuwat Tangthanawatsakul

A report for the course **MTH 234 Linear Algebra**  
King Mongkut’s University of Technology Thonburi  
Semester 1, Academic Year 2025

---

## Abstract

The Hessian matrix **H** is a square symmetric matrix of second-order partial derivatives of a multivariable scalar function \(f(x)\). It serves as a fundamental tool for analyzing local curvature in multidimensional spaces. The matrix is used as the core component for approximating a function around a critical point using a quadratic form through the Taylor expansion.

The mathematical characteristics of H are crucial due to the following reasons:

1. **Symmetry** ensures that eigenvalues are real and allows stable inversion of the matrix \(H^{-1}\).
2. **Eigenvectors** represent principal curvature directions, and **eigenvalues** indicate curvature magnitude, enabling the construction of an _orthonormal basis_ for deeper analysis.
3. The matrix–vector product \(Hv\) directly measures the effect of curvature along direction \(v\).

These components assist in identifying critical points (local minima, maxima, saddle points) and form the foundation of second-order algorithms such as Newton’s Method and Quasi-Newton Methods.

---

## Introduction

Linear Algebra is a core foundation of modern computation, playing a vital role in multidimensional data processing, signal computation, large-scale data analytics, and algorithm design. Deep Learning in particular heavily relies on vector and matrix structures and linear transformations.

One prominent topic is the analysis of the curvature of a loss function through the _Hessian Matrix_. This matrix captures the curvature structure of the loss landscape, enabling identification of minima, saddle points, and maxima, and serves as the basis for second-order optimization.

This report explains the role of Linear Algebra in Deep Learning and the significance of the Hessian in analyzing and improving training performance.

---

## Mathematical Concepts from Linear Algebra

- **Matrix-Vector Product**: Used to evaluate the effect of \(H\) on a vector \(v\) without explicitly forming H
- **Orthogonal / Orthonormal Basis**: Essential for eigen decomposition
- **Eigenvalues / Eigenvectors**: Describe directions and magnitudes of curvature
- **Eigen Basis**: A basis constructed from eigenvectors
- **Hessian Matrix (H)**: A symmetric matrix of second-order derivatives
- **Linear Transformation**: Viewing the Hessian as a linear operator
- **Taylor Series**: Used to approximate the loss with a quadratic form

---

## Detail Description

### 1. Principles of Deep Learning

#### 1.1 Neural Networks and Non-linear Relations

Explanation of layer structure, activation functions, and non-linear relationships.

#### 1.2 Forward Pass

Basic equations:

\[
z^{(l)} = a^{(l-1)}W^{(l)} + b^{(l)}
\]
\[
a^{(l)} = f(z^{(l)})
\]

#### Loss Computation

Using Mean Square Error:

\[
L = \frac{1}{N} \sum_i (y_i - \hat{y}\_i)^2
\]

---

### 2. Gradient via First-Order Derivatives

Parameter updates using gradient descent:

\[
W \leftarrow W - \eta \nabla_W L
\]

Using the chain rule to compute the backward pass (backpropagation).

---

### 3. Role of the Hessian in Deep Learning

#### 3.1 Definition and Dimension of the Hessian

\(
\nabla L(\theta) \in \mathbb{R}^p
\)  
\(
H = \nabla^2 L(\theta) \in \mathbb{R}^{p \times p}
\)

#### 3.2 Key Properties of the Hessian

- **Clairaut's Theorem**: mixed partial derivatives are equal → H is symmetric
- **Spectral Theorem**:

\[
H = Q \Lambda Q^T
\]

where \(Q\) is orthogonal and \(\Lambda\) is diagonal.

#### 3.3 Using the Hessian in Taylor Series

\[
L(\theta + \Delta) \approx
L(\theta) + \nabla L^T \Delta + \frac{1}{2}\Delta^T H \Delta
\]

Used to analyze curvature and determine optimal update directions.

---

## Application Example: Deep Autoencoder

An example using Hessian-Free Optimization (HF).

### Main Steps

1. Form the linear system:
   \[
   H p = -\nabla L
   \]
2. Use Conjugate Gradient (CG) to solve it without forming H
3. Use Hessian-vector products \(Hv\)
4. Update parameters using the direction \(p\)

Result: Converges faster than SGD, especially in ill-conditioned regions.

---

## Discussion and Analysis

### Advantages

- Effective in areas with complex curvature
- Requires fewer epochs

### Limitations

- H may not be positive-definite
- Higher computational cost than first-order methods
- CG requires tuning

---

## Conclusion

The Hessian matrix provides insight into the curvature of the loss landscape and enhances training efficiency through second-order methods. Although models can be large, techniques such as Hessian-vector products, FIM, and K-FAC make second-order optimization feasible in practice.

---

## References

- Frans, K. (2023). _Second-Order Optimization_. Notes on Deep Learning.
- KMUTT. (n.d.). _MTH 234 Course Material_.
- Martens, J. (2010). _Deep learning via Hessian-free optimization_.
