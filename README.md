# npMF

NumPy Matrix Factorization (npMF) is a Python package that only depends on NumPy providing a unified interface to different constrained and unconstrained low-rank matrix factorization methods.

npMF currently implements the following algorithms:
- Stochastic gradient descent (SGD), with and without biases
- Alternating least squares (ALS), with and without biases
- Alternating nonnegative least squares (ANLS)
- [Bounded matrix factorization](https://doi.org/10.1007/s10115-013-0710-2) (BMF)

Each of these methods is also extended to take a matrix of confidence levels to give each entry a different weight, as commonly required in implicit feedback recommender systems.

A few initialization, quality scoring and learning rate decay functions are also included in npMF.


## Usage

### Using a model

Given a user-item ratings matrix 
<a href="https://www.codecogs.com/eqnedit.php?latex=\textbf{M}&space;\in&space;R^{D\times&space;N}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\textbf{M}&space;\in&space;R^{D\times&space;N}" title="\textbf{M} \in R^{D\times N}" /></a>
whose missing entries are denoted by 0s, we can find a rank-*k* approximation given by 
<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{\textbf{M}}&space;=&space;\textbf{W}&space;\cdot&space;\textbf{Z}^T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{\textbf{M}}&space;=&space;\textbf{W}&space;\cdot&space;\textbf{Z}^T" title="\hat{\textbf{M}} = \textbf{W} \cdot \textbf{Z}^T" /></a>
, where 
<a href="https://www.codecogs.com/eqnedit.php?latex=\textbf{W}&space;\in&space;R^{D\times&space;k}&space;\text{&space;and&space;}&space;\textbf{Z}&space;\in&space;R^{N\times&space;k}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\textbf{W}&space;\in&space;R^{D\times&space;k}&space;\text{&space;and&space;}&space;\textbf{Z}&space;\in&space;R^{N\times&space;k}" title="\textbf{W} \in R^{D\times k} \text{ and } \textbf{Z} \in R^{N\times k}" /></a>
, by using SGD as follows:

```python
import npmf.models
from npmf.learning_rate_decay import inverse_time_decay

# data
M = ...

# hyperparameters
k = 3
init_lr = 0.1
decay_rate = 1/1.2
lambda_u = 0.1
lambda_i = 0.1
nanvalue = 0
max_iter=2000

# factorize data matrix
W, Z, user_biases, item_biases, err_train, pred_fn = \
    npmf.models.sgd(M, num_features=k, nanvalue=nanvalue, lr0=init_lr,
                    decay_fn=lambda lr, step: inverse_time_decay(lr, step, decay_rate, max_iter),
                    lambda_user=lambda_u, lambda_item=lambda_i, max_iter=max_iter)
M_hat = pred_fn(W, Z, user_biases, item_biases)
```

### Using a weighted model
If we also have access to a confidence matrix 
<a href="https://www.codecogs.com/eqnedit.php?latex=\textbf{C}&space;\in&space;R^{D\times&space;N}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\textbf{C}&space;\in&space;R^{D\times&space;N}" title="\textbf{C} \in R^{D\times N}" /></a>
giving different weights to each observed entry in *M*, we can find a rank-*k* approximation given by 
<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{\textbf{M}}&space;=&space;\tilde{\textbf{W}}\cdot&space;\tilde{\textbf{Z}}^T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{\textbf{M}}&space;=&space;\tilde{\textbf{W}}\cdot&space;\tilde{\textbf{Z}}^T" title="\hat{\textbf{M}} = \tilde{\textbf{W}}\cdot \tilde{\textbf{Z}}^T" /></a>
, where
<a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{\textbf{W}}&space;=&space;[\boldsymbol{\beta}&space;~~&space;\textbf{W}]&space;\in&space;R^{D\times&space;k&plus;1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tilde{\textbf{W}}&space;=&space;[\boldsymbol{\beta}&space;~~&space;\textbf{W}]&space;\in&space;R^{D\times&space;k&plus;1}" title="\tilde{\textbf{W}} = [\boldsymbol{\beta} ~~ \textbf{W}] \in R^{D\times k+1}" /></a>
is obtained by prepending the vector of user biases to
<a href="https://www.codecogs.com/eqnedit.php?latex=\textbf{W}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\textbf{W}" title="\textbf{W}" /></a>
(analogously for the other factor), by using SGD as follows:

```python
from npmf.weighted_models import sgd_bias_weight
from npmf.learning_rate_decay import exponential_decay

# data
M = ...

# hyperparameters
k = 3
init_lr = 0.1
decay_rate = 1/1.2
lambda_u = 0.1
lambda_i = 0.1
nanvalue = 0
max_iter=2000

# factorize data matrix
W, Z, user_biases, item_biases, err_train, pred_fn = \
    sgd_bias_weight(M, confidence=C, num_features=k, nanvalue=nanvalue, lr0=init_lr,
                    decay_fn=lambda lr, step: exponential_decay(lr, step, decay_rate, max_iter),
                    lambda_user=lambda_u, lambda_item=lambda_i, max_iter=max_iter)
M_hat = pred_fn(W, Z, user_biases, item_biases)
```

### Using a class for a model
We can keep track of every parameter as well as different errors for a model into a class as follows:

```python
import npmf.models
import npmf.error_metrics
from npmf.learning_rate_decay import inverse_time_decay
from npmf.wrapper_classes import MF

# data
train_matrix = ...
valid_matrix = ...
test_matrix = ...

# hyperparameters
k = 3
init_lr = 0.1
decay_rate = 1/1.2
lambda_u = 0.1
lambda_i = 0.1
nanvalue = 0
max_iter=2000

# instantiate model class
SGD = MF(npmf.models.sgd, num_features=k, nanvalue=nanvalue, lr0=init_lr,
         decay_fn=lambda lr, step: inverse_time_decay(lr, step, decay_rate, max_iter, False),
         lambda_user=lambda_u, lambda_item=lambda_i, max_iter=max_iter)

# train
SGD.fit(train_matrix)
# predict
predicted_matrix = SGD.predict()
# retrieve factors
W = SGD.user_features 
Z = SGD.item_features
# evaluate scores
SGD.score(err_fn=npmf.error_metrics.rmse, matrix=train_matrix, err_type='train')
SGD.score(err_fn=npmf.error_metrics.rmse, matrix=valid_matrix, err_type='validation')
SGD.score(err_fn=npmf.error_metrics.rmse, matrix=test_matrix, err_type='test')

print(SGD.train_errors)
```

### Using a class for cross-validating a model
We can use a cross-validation class to keep track of every parameter of a model and evaluate its performance over multiple splits:

```python
import npmf.models
import npmf.error_metrics
from npmf.learning_rate_decay import inverse_time_decay
from npmf.wrapper_classes import CvMF

# data
train_matrices = [...]
valid_matrices = [...]

# hyperparameters
k = 3
init_lr = 0.1
decay_rate = 1/1.2
lambda_u = 0.1
lambda_i = 0.1
nanvalue = 0
max_iter=2000

# instantiate model class
cvSGD = CvMF(npmf.models.sgd_bias, num_features=k, nanvalue=nanvalue, lr0=init_lr,
             decay_fn=lambda lr, step: inverse_time_decay(lr, step, decay_rate, max_iter, False),
             lambda_user=lambda_u, lambda_item=lambda_i, max_iter=max_iter)

# fit the model
cvSGD.fit(train_matrices)
# training accuracy
SGD.score(err_fn=npmf.error_metrics.rmse, matrix=train_matrices, err_type='train', 
          agg_fn=np.mean, dev_fn=npmf.error_metrics.se)
SGD.score(err_fn=npmf.error_metrics.mae, matrix=train_matrices, err_type='train', 
          agg_fn=np.mean, dev_fn=npmf.error_metrics.se)
# validation accuracy
SGD.score(err_fn=npmf.error_metrics.rmse, matrix=valid_matrices, err_type='validation', 
          agg_fn=np.mean, dev_fn=npmf.error_metrics.se)
SGD.score(err_fn=npmf.error_metrics.mae, matrix=valid_matrices, err_type='validation', 
          agg_fn=np.mean, dev_fn=npmf.error_metrics.se)
```
## License

The project was started at [Technicolor AI Lab](http://www.technicolorbayarea.com/) in early 2018.
It is now distributed under the MIT license.
