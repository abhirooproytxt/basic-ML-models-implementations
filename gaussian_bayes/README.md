Gaussian Bayes Classification (QDA) (No Naive Assumptions)
<br>
<br>

Assumptions:

1. Features follow multivariate Gaussians for each class.

$$x \mid y = k \sim \mathcal{N}(\mu_k, \Sigma_k)$$

$$P(x|y=k)=\frac{1}{(2\pi)^{d/2}|\Sigma_k|^{1/2}}
\exp\left(-\frac{1}{2}(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)\right)
$$


2. Class-specific means and covariance matrices. $$\mu_k, \Sigma_k$$  Enough data belonging to each class to calculate means and covariances for each class, not shared ones. This makes it a Quadratic Discriminant Analysis (QDA). The decision boundaries are quadratic.



3. Class priors exist. $$ P(y=k) $$ 
Class prior probability exist for a class k.
Total number of classes in training data is known.

<br>
<br>

Classification:

Posterior probability, 

$$
P(y=k \mid x)
\propto
P(x \mid y=k)P(y=k)
$$

$$
P(y=k \mid x)=
\frac{
P(x \mid y=k)P(y=k)
}{
\sum_{j=1}^{K} P(x \mid y=j)P(y=j)
}
$$

We ignore the marginal probability, because it is a constant. And we use log scores to prevent comparing too small values.
So prediction,

$$\hat{c} = \arg\max_{c} \, \left[ \log P(\mathbf{x} \mid c) + \log P(c) \right]$$


