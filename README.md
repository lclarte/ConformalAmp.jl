This repository contains the Julia code used to reproduce the experiments of the paper "Building Conformal Prediction Intervals with Approximate Message Passing". The dependencies are listed in the `Project.toml` file.

# Reproducing the experiments 

* Running the scripts `experiments/paper/boston.jl` and `experiments/paper/riboflavin.jl` reproduce the results of Table 4 for real data in the paper. For a comparison with approximate homotopy, you can run the files `python/homotopy_boston.py` and `python/homotopy_riboflavin.py`. 

# General structure 

The code is structured as follows:

* `src`: contains the code for AMP and Taylor-AMP described in the paper. In particular, the functions `gamp` and `compute_order_one_perturbation_gamp` in the file `gamp.jl` respectively implement the algorithms AMP and Taylor-AMP described in the paper.
* `experiments` : contains various experiments, the sub-folder `paper` contains the script used to produce the plots of the paper. The folder `misc` contains additional (undocumented) experiments. The folder `python` contains Python code to compare with other methods implemented in Python.

### External libraries 

* The folder `homotopy_conformal_prediction` is from the repository [_Computing Full Conformal Prediction Set with Approximate Homotopy_](http://github.com/EugeneNdiaye/homotopy_conformal_prediction)

# Usage

Consider training dataset `X, y` and a test vector `xtest`. The following code snippet shows how to compute confidence intervals for Full Conformal Prediction for the Ridge regression case:

```
using ConformalAmp

# parameters α and Δ are not used in the computations yet 
problem = ConformalAmp.Ridge(Δ = 1.0, Δ̂ = 1.0, λ = 1.0, α = 1.0)
# alternatively do Lasso regression
problem = ConformalAmp.Lasso(Δ = 1.0, Δ̂ = 1.0, λ = 1.0, α = 1.0)

# define the method to use : either full or split conformal prediction
conformal = ConformalAmp.FullConformal(coverage = coverage, δy_range = 0.0:0.05:5.0)
algo      = ConformalAmp.GAMPTaylor(max_iter = 100, rtol = 1e-4)
# algo      = ConformalAmp.GAMP(max_iter = 100, rtol = 1e-4)

interval = ConformalAmp.get_confidence_interval(problem, X, y, xtest, algo, method)
```

The parameters Δ, α are only used to generate synthetic datasets and are not required to compute estimators, Δ̂ is an estimation of the Gaussian noise in the square loss (we fix it at 1.0 in the paper) and λ is the regularisation level. To compute the confidence intervals exactly with our method, replace the line `algo = ConformalAmp.GAMPTaylor(max_iter = 100, rtol = 1e-4)` with `algo = ConformalAmp.ERM()`.

Note that the input `X` of size `n \times d` should be centered (0 mean) and have scale `~ 1 / sqrt(d)` for the algorithm to converge.

## Additional details

Unless otherwise specified, all confidence intervals in the numerics of the paper are given for a confidence level of 90%. In addition to the Ridge and Lasso regression cases studied in the paper, this code is able to compute leave-one-out estimators for the Logistic and Pinball (quantile regression) cases.
