This repository contains the Julia code used to reproduce the experiments of the paper "Building Conformal Prediction Intervals with Approximate Message Passing". The dependencies are listed in the `Project.toml` file.

# General structure 

The code is structured as follows:

<<<<<<< HEAD
-`src`: contains the code for AMP and Taylor-AMP described in the paper, especially the file `gamp.jl`
- `experiments` : contains various experiments, the sub-folder 
=======
* `src`: contains the code for AMP and Taylor-AMP described in the paper. In particular, the functions `gamp` and `compute_order_one_perturbation_gamp` in the file `gamp.jl` respectively implement the algorithms AMP and Taylor-AMP described in the paper.
* `experiments` : contains various experiments, the sub-folder `paper` contains the script used to produce the plots of the paper. The folder `misc` contains additional (undocumented) experiments.

# Usage

Consider training dataset `X, y` and a test vector `xtest`. The following code snippet shows how to compute confidence intervals for Full Conformal Prediction for the Ridge regression case:

```
using ConformalAmp

problem = ConformalAmp.Ridge(Δ = 1.0, Δ̂ = 1.0, λ = 1.0, α = 1.0)
# alternatively do Lasso regression

# define the method to use : either full or split conformal prediction
conformal = ConformalAmp.FullConformal(coverage = coverage, δy_range = -0.0:0.05:5.0)
algo      = ConformalAmp.GAMPTaylor(max_iter = 100, rtol = 1e-4)

interval = ConformalAmp.get_confidence_interval(problem, X, y, xtest, algo, method)
```

The parameters Δ, α are only used to generate synthetic datasets and are not required to compute estimators, Δ̂ is an estimation of the Gaussian noise in the square loss (we fix it at one in the paper) and λ is the regularisation level. To compute the confidence intervals exactly with our method, replace the line `algo = ConformalAmp.GAMPTaylor(max_iter = 100, rtol = 1e-4)` with `algo = ConformalAmp.ERM()`.

## Additional details

Unless otherwise specified, all confidence intervals in the numerics of the paper are given for a confidence level of 90%. In addition to the Ridge and Lasso regression cases studied in the paper, this code is able to compute leave-one-out estimators for the Logistic and Pinball (quantile regression) cases.
>>>>>>> clean
