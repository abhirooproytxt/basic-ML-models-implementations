# Logistic Regression from Scratch (using MAP)

This project implements **Binary Logistic Regression** from scratch using **Maximum A Posteriori (MAP) Estimation** with **L2 Regularization**.

## Model

For labels (y_i \in {-1,+1}), the probability of correct classification is:

$$
P(y_i \mid x_i,w)=\sigma(y_i x_i^T w)
$$

where

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

---

## MAP Objective

A Gaussian prior is placed on the weights, resulting in the following objective function:

$$
J(w)
=

\sum_{i=1}^{n}
\log\left(\sigma(y_i x_i^T w)\right)
-

\frac{\lambda}{2}w^T w
$$

The first term maximizes the likelihood of the data, while the second term performs L2 regularization.

---

## Gradient

The gradient of the objective is:

$$
\nabla J(w)
=

\sum_{i=1}^{n}
\left(1-\sigma(y_i x_i^T w)\right)
y_i x_i
-

\lambda w
$$

Vectorized form:

$$
\nabla J(w)
=

X^T
\left(
y \odot
\left(
1-\sigma(y\odot(Xw))
\right)
\right)
-

\lambda w
$$

---

## Optimization

Weights are learned using gradient ascent:

$$
w \leftarrow w + \eta \nabla J(w)
$$

where eta is the learning rate.

---

## Prediction

For a test sample \(x\), the predicted probability is

$$
P(y=1|x)=\sigma(x^T w)
$$


The predicted class is obtained using a threshold of \(0.5\):

$$
\hat y=
\begin{cases}
+1 & \sigma(x^T w)\ge 0.5\\
-1 & \sigma(x^T w)<0.5
\end{cases}
$$


## Features

* Binary Logistic Regression
* MAP Estimation
* L2 Regularization
* Fully Vectorized NumPy Implementation
* Gradient Ascent Optimization
* PCA-based Visualization
