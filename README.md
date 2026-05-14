# Fourier shaped neural net

![Alt text here](images/fourier-neural-net.jpeg)

## Model

A periodic function $f(t)$ with period $T$, i.e. $f(t+T)=f(t)$, can be approximated, under mild conditions, as a sum of trigonometric functions:

$$
f(t) \approx S_n(t) = c_0 + 
\sum_{j=1}^n 
    a_j \sin \left( \frac{2\pi j}{T}t + \phi_j \right).
$$

Here we address the regression problem whereby $f$ can also depend on a collection of covariates $\mathbf{x}=(x_1,\ldots, x_k)$

$$
f(t;\mathbf{x}) \approx S_n(t;\mathbf{x}) = c(\mathbf{x}) + 
\sum_{j=1}^n a_j(\mathbf{x}) \sin \left( \frac{2\pi j}{T}t + \phi_j(\mathbf{x}) \right).
$$

Given a set of harmonics $\mathcal{J}$, a known period $T$, and a training dataset $\mathcal{D}$ consisting of timepoints, covariate values and responses

$$
\mathcal{D} = \{t_i, \mathbf{x}_i=(x_{i,1}, \ldots, x_{i, k}), y_i \;|\; i=1,\ldots,N \}
$$

the model trains a feed-forward neural network $\Phi$ representing the Fourier coefficients as a function of the covariates:

$$
\Phi(\mathbf{x}) = (c_0(\mathbf{x}), a_j(\mathbf{x}), \phi_j(\mathbf{x}))_{j\in\mathcal{J}}
$$

so that it minimizes the loss

$$
\sum_{i=1}^N \left[S_n(t_i; \mathbf{x}_i) - y_i\right]^2.
$$