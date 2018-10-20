# -*- coding: utf-8 -*-
"""
Created by e-bug on 7/17/18
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from npmf.error_metrics import *
from npmf.learning_rate_decay import *
from npmf.init_functions import *
from npmf.utils import *

import numpy as np


# ==================================================================================================================== #
#                                                                                                                      #
#                                  Matrix Factorization models with confidence levels                                  #
#                                                                                                                      #
# ==================================================================================================================== #

def sgd_bias_weight(train, confidence, init_fn=rand_init, num_features=6, nanvalue=0,
                    lr0=0.01, decay_fn=lambda lr, step: inverse_time_decay(lr, step, 0.5, 2000, False), batch_size=32,
                    lambda_user=0.1, lambda_item=0.1, max_epochs=2000, stop_criterion=1e-6,
                    err_fn=rmse, display=50, seed=42, **kwargs):
    """
    SGD (Stochastic Gradient Descent) weighted by confidence levels for low-rank matrix factorization with biases.
    
    Args:
        train: Ratings matrix to be factorized
        confidence: Matrix of number of observations resulting in entries of `train`
        init_fn: Function to initialize factor matrices
        num_features: Number of latent factors to be used in factorizing `train`
        nanvalue: Value used in `train` indicating a missing entry
        lr0: Intial learning rate
        decay_fn: Learning rate decay function. If None, keeps it constant
        batch_size: Number of samples employed for each training step
        lambda_user: Regularization strength for users' parameters
        lambda_item: Regularization strength for item' parameters
        max_epochs: Maximum number of epochs
        stop_criterion: Minimum relative difference in loss function to continue training
        err_fn: Function to evaluate training performance
        display: Interval of number of epochs after which to print progress
        seed: Random seed
    Returns:
        Factor matrix of users, factor matrix of items, users' biases, items' biases, 
        final training error, function to compute prediction matrix
    """

    # define parameters
    change = 1
    loss_list = [np.finfo(np.float64).max]
    err_list = []
    if decay_fn is None:
        decay_fn = lambda lr, step: lr

    # set seed
    np.random.seed(seed)

    # define function to predict matrix
    pred_fn = lambda user_feats, item_feats, user_bias, item_bias: \
        user_feats.dot(item_feats.T) + user_bias[:, None] + item_bias

    # init matrix
    user_features, item_features = init_fn(train.shape[0], train.shape[1], num_features)
    user_biases, item_biases = np.zeros(user_features.shape[0]), np.zeros(item_features.shape[0])
    R = 1 + 0.1 * np.log(1 + confidence)

    # find the non-zero ratings indices
    nz_row, nz_col = np.argwhere(train != nanvalue)[:, 0], np.argwhere(train != nanvalue)[:, 1]
    nz_train = list(zip(nz_row, nz_col))
    O = train != nanvalue
    num_nz = np.sum(O)

    # run
    print("start Weighted SGD...")
    e = 0
    while change > stop_criterion and e < max_epochs:

        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        batches = get_batches(nz_train, batch_size)

        # decrease step size
        lr = decay_fn(lr0, e)

        for b in batches:
            mask = np.zeros_like(train)
            mask[b[:, 0], b[:, 1]] = 1
            errs = mask * (train - pred_fn(user_features, item_features, user_biases, item_biases))

            # update user_features and item_features
            user_feats = user_features.copy()
            user_features += lr * ((R * errs.dot(item_features))/batch_size - lambda_user * user_features)
            item_features += lr * ((R * (errs.T).dot(user_feats))/batch_size - lambda_item * item_features)
            user_biases += lr * (np.sum(R * errs, axis=1)/batch_size - lambda_user * user_biases)
            item_biases += lr * (np.sum(R * errs, axis=0)/batch_size - lambda_item * item_biases)

        # train error
        P = pred_fn(user_features, item_features, user_biases, item_biases)
        err = err_fn(train, P, O)

        # loss value
        loss = 0.5 * (np.sum(R * np.power(O * (train - P), 2))/num_nz
                      + lambda_user * (np.sum(np.power(user_features, 2)) + np.sum(np.power(user_biases, 2)))
                      + lambda_item * (np.sum(np.power(item_features, 2)) + np.sum(np.power(item_biases, 2))))

        if display and not e % display:
            print("epoch: {:4d}, loss: {:e} -- {} on training set: {:e} .".format(e, loss, err_fn.__name__, err))
        loss_list.append(loss)
        err_list.append(err)
        change = np.fabs(loss_list[-1] - loss_list[-2]) / np.fabs(loss_list[-1])
        e += 1
    print("epoch: {:4d}, loss: {:e} -- {} on training set: {:e} .".format(e, loss, err_fn.__name__, err))

    return user_features, item_features, user_biases, item_biases, loss_list[1:], err_list, pred_fn


def als_bias_weight(train, confidence, init_fn=rand_init, num_features=6, nanvalue=0,
                    lambda_user=0.1, lambda_item=0.1, max_epochs=2000, stop_criterion=1e-6,
                    err_fn=rmse, display=50, seed=42, **kwargs):
    """
    ALS (Alternating Least Squares) weighted by confidence levels for low-rank matrix factorization with biases.
    
    Args:
        train: Ratings matrix to be factorized
        confidence: Matrix of number of observations resulting in entries of `train`
        init_fn: Function to initialize factor matrices
        num_features: Number of latent factors to be used in factorizing `train`
        nanvalue: Value used in `train` indicating a missing entry
        lambda_user: Regularization strength for users' parameters
        lambda_item: Regularization strength for item' parameters
        max_epochs: Maximum number of epochs
        stop_criterion: Minimum relative difference in loss function to continue training
        err_fn: Function to evaluate training performance
        display: Interval of number of epochs after which to print progress
        seed: Random seed
    Returns:
        Factor matrix of users, factor matrix of items, users' biases, items' biases, 
        final training error, function to compute prediction matrix
    """

    # define parameters
    change = 1
    loss_list = [np.finfo(np.float64).max]
    err_list = []

    # set seed
    np.random.seed(seed)

    # define function to predict matrix
    pred_fn = lambda user_feats, item_feats, user_bias, item_bias: \
        user_feats.dot(item_feats.T) + user_bias[:, None] + item_bias

    # init matrix
    user_features, item_features = init_fn(train.shape[0], train.shape[1], num_features)
    user_biases, item_biases = np.zeros(user_features.shape[0]), np.zeros(item_features.shape[0])
    R = np.sqrt(1 + 0.1 * np.log(1 + confidence))

    # find the non-zero ratings indices
    O = train != nanvalue
    num_nz = np.sum(O)

    # run
    print("start Weighted ALS...")
    e = 0
    while change > stop_criterion and e < max_epochs:
        # update user feature
        for d in range(train.shape[0]):
            nz_entries = np.argwhere(train[d, :] != nanvalue).T[0]
            Z = item_features[nz_entries, :]
            Z_tilde = np.concatenate((np.ones((Z.shape[0], 1)), Z), axis=1)
            r = R[d, nz_entries]
            m = (Z_tilde.T).dot(np.diag(r))
            A = m.dot(r[:, np.newaxis] * Z_tilde) + lambda_user * num_nz * np.identity(num_features+1)
            b = m.dot(train[d, nz_entries] - item_biases[nz_entries])
            user_upd = np.linalg.solve(A, b)
            user_biases[d] = user_upd[0]
            user_features[d, :] = user_upd[1:]

        # update item feature
        for n in range(train.shape[1]):
            nz_entries = np.argwhere(train[:, n] != nanvalue).T[0]
            W = user_features[nz_entries, :]
            W_tilde = np.concatenate((np.ones((W.shape[0], 1)), W), axis=1)
            r = R[nz_entries, n]
            m = (W_tilde.T).dot(np.diag(r))
            A = m.dot(r[:, np.newaxis] * W_tilde) + lambda_item * num_nz * np.identity(num_features+1)
            b = m.dot(train[nz_entries, n] - user_biases[nz_entries])
            item_upd = np.linalg.solve(A, b)
            item_biases[n] = item_upd[0]
            item_features[n, :] = item_upd[1:]

        # train error
        P = pred_fn(user_features, item_features, user_biases, item_biases)
        err = err_fn(train, P, O)

        # loss value
        loss = 0.5 * (np.sum(R**2 * np.power(O * (train - P), 2))/num_nz
                      + lambda_user * (np.sum(np.power(user_features, 2)) + np.sum(np.power(user_biases, 2)))
                      + lambda_item * (np.sum(np.power(item_features, 2)) + np.sum(np.power(item_biases, 2))))

        if display and not e % display:
            print("epoch: {:4d}, loss: {:e} -- {} on training set: {:e} .".format(e, loss, err_fn.__name__, err))
        loss_list.append(loss)
        err_list.append(err)
        change = np.fabs(loss_list[-1] - loss_list[-2]) / np.fabs(loss_list[-1])
        e += 1
    print("epoch: {:4d}, loss: {:e} -- {} on training set: {:e} .".format(e, loss, err_fn.__name__, err))

    return user_features, item_features, user_biases, item_biases, loss_list[1:], err_list, pred_fn


def anls_weight(train, confidence, init_fn=rand_init, num_features=6, nanvalue=0,
                lr0=0.01, decay_fn=lambda lr, step: inverse_time_decay(lr, step, 0.5, 2000, False), 
                lambda_user=0.1, lambda_item=0.1, max_epochs=2000, int_iter=200, stop_criterion=1e-6,
                err_fn=rmse, display=50, seed=42, **kwargs):
    """
    ANLS (Alternating Nonnegative Least Squares) weightd by confidence levels for nonnegative matrix factorization.
    
    Args:
        train: Ratings matrix to be factorized
        confidence: Matrix of number of observations resulting in entries of `train`
        init_fn: Function to initialize factor matrices
        num_features: Number of latent factors to be used in factorizing `train`
        nanvalue: Value used in `train` indicating a missing entry
        lr0: Initial learning rate
        decay_fn: Learning rate decay function. If None, keeps it constant
        lambda_user: Regularization strength for users' parameters
        lambda_item: Regularization strength for item' parameters
        max_epochs: Maximum number of epochs
        int_iter: Maximum number of iterations in inner iterative procedures
        stop_criterion: Minimum relative difference in loss function to continue training
        err_fn: Function to evaluate training performance
        display: Interval of number of epochs after which to print progress
        seed: Random seed
    Returns:
        Factor matrix of users, factor matrix of items, None, None, 
        final training error, function to compute prediction matrix
    """

    # define parameters
    change = 1
    loss_list = [np.finfo(np.float64).max]
    err_list = []
    if decay_fn is None:
        decay_fn = lambda lr, step: lr

    # set seed
    np.random.seed(seed)

    # define function to predict matrix
    pred_fn = lambda user_feats, item_feats, user_bias, item_bias: user_feats.dot(item_feats.T)

    # init matrix
    user_features, item_features = init_fn(train.shape[0], train.shape[1], num_features)

    # find the non-zero ratings indices
    O = train != nanvalue
    R = 1 + 0.1 * np.log(1 + confidence)
    M = O * R
    num_nz = np.sum(O)

    # run
    print("start Weighted ANLS...")
    e = 0
    while change > stop_criterion and e < max_epochs:

        # decrease step size
        lr = decay_fn(lr0, e)

        # update user feature
        int_it = 0
        int_change = 1
        int_loss_list = [np.finfo(np.float64).max]
        P = pred_fn(user_features, item_features, None, None)
        while int_change > stop_criterion and int_it < int_iter:
            user_features += lr * ((M * (train -P)).dot(item_features)/num_nz - lambda_user * user_features)
            user_features = np.maximum(user_features, 0)
            P = pred_fn(user_features, item_features, None, None)
            loss = 0.5 * (np.sum(M * np.power(train - P, 2))/num_nz
                          + lambda_user * np.sum(np.power(user_features, 2))
                          + lambda_item * np.sum(np.power(item_features, 2)))
            int_loss_list.append(loss)
            int_change = np.fabs(int_loss_list[-1] - int_loss_list[-2]) / np.fabs(int_loss_list[-1])
            int_it += 1

        # update item feature
        int_it = 0
        int_change = 1
        int_loss_list = [np.finfo(np.float64).max]
        while int_change > stop_criterion and int_it < int_iter:
            item_features += lr * (((M * (train - P)).T).dot(user_features)/num_nz - lambda_item * item_features)
            item_features = np.maximum(item_features, 0)
            P = pred_fn(user_features, item_features, None, None)
            loss = 0.5 * (np.sum(M * np.power(train - P, 2))/num_nz
                          + lambda_user * np.sum(np.power(user_features, 2))
                          + lambda_item * np.sum(np.power(item_features, 2)))
            int_loss_list.append(loss)
            int_change = np.fabs(int_loss_list[-1] - int_loss_list[-2]) / np.fabs(int_loss_list[-1])
            int_it += 1

        # train error
        err = err_fn(train, P, O)

        # loss value
        loss = 0.5 * (np.sum(M * np.power(train - P, 2))/num_nz
                      + lambda_user * np.sum(np.power(user_features, 2))
                      + lambda_item * np.sum(np.power(item_features, 2)))

        if display and not e % display:
            print("epoch: {:4d}, loss: {:e} -- {} on training set: {:e} .".format(e, loss, err_fn.__name__, err))
        loss_list.append(loss)
        err_list.append(err)
        change = np.fabs(loss_list[-1] - loss_list[-2]) / np.fabs(loss_list[-1])
        e += 1
    print("epoch: {:4d}, loss: {:e} -- {} on training set: {:e} .".format(e, loss, err_fn.__name__, err))

    return user_features, item_features, None, None, loss_list[1:], err_list, pred_fn


def bmf_weight(train, confidence, init_fn=rand_init, num_features=6, nanvalue=0,
               xmin=0, xmax=1, max_epochs=2000, stop_criterion=1e-6,
               err_fn=rmse, display=50, seed=42, **kwargs):
    """
    BMF (Bounded Matrix Factorization) weighted by confidence levels for bounded matrix factorization.
    
    Args:
        train: Ratings matrix to be factorized
        confidence: Matrix of number of observations resulting in entries of `train`
        init_fn: Function to initialize factor matrices
        num_features: Number of latent factors to be used in factorizing `train`
        nanvalue: Value used in `train` indicating a missing entry
        xmin: Minimum value that can be predicted 
        xmax: Maximum value that can be predicted
        max_epochs: Maximum number of epochs
        stop_criterion: Minimum relative difference in loss function to continue training
        err_fn: Function to evaluate training performance
        display: Interval of number of epochs after which to print progress
        seed: Random seed
    Returns:
        Factor matrix of users, factor matrix of items, None, None, 
        final training error, function to compute prediction matrix
    """

    def lower_bounds(x_min, x_max, t, v):
        b = -np.infty * np.ones(v.shape[0])
        for j in range(v.shape[0]):
            if v[j] > 0:
                b[j] = (x_min - t[j]) / v[j]
            elif v[j] < 0:
                b[j] = (x_max - t[j]) / v[j]
        return b

    def upper_bounds(x_min, x_max, t, v):
        b = +np.infty * np.ones(v.shape[0])
        for j in range(v.shape[0]):
            if v[j] > 0:
                b[j] = (x_max - t[j]) / v[j]
            elif v[j] < 0:
                b[j] = (x_min - t[j]) / v[j]
        return b

    def find_element(v, x, o, t, l_max, u_min):
        e = ((o * (x - t)).T).dot(v) / (np.power(np.linalg.norm(o * v), 2))
        if e < l_max:
            e = l_max
        elif e > u_min:
            e = u_min
        return e

    # define parameters
    change = 1
    loss_list = [np.finfo(np.float64).max]
    err_list = []
    D, N = train.shape

    # set seed
    np.random.seed(seed)

    # define function to predict matrix
    pred_fn = lambda user_feats, item_feats, user_bias, item_bias: user_feats.dot(item_feats.T)

    # init matrix
    user_features, item_features = init_fn(train.shape[0], train.shape[1], num_features)

    # find the non-zero ratings indices
    O = train != nanvalue
    R = np.sqrt(1 + 0.1 * np.log(1 + confidence))
    M = O * R
    num_nz = np.sum(O)

    # init BMF
    if xmax == 1:
        max_el = user_features.dot(item_features.T).max()
        user_features /= np.sqrt(max_el)
        item_features /= np.sqrt(max_el)
    elif xmax > 1:
        max_el = (user_features[:, 1:]).dot(item_features[:, 1:].T).max()
        alpha = np.sqrt((xmax - 1) / max_el)
        user_features *= alpha
        item_features *= alpha
        user_features[:, 0] = 1
        item_features[:, 0] = 1
    else:
        raise ValueError('Invalid maximum data matrix value')

    # run
    print("start Weighted BMF...")
    e = 0
    best_W, best_Z = None, None
    min_l = np.finfo(np.float64).max
    while change > stop_criterion and e < max_epochs:

        for k in range(num_features):
            T = user_features.dot(item_features.T) - user_features[:, k, None].dot(item_features[:, k, None].T)
            for d in range(D):
                l = lower_bounds(xmin, xmax, T[d, :], item_features[:, k])
                u = upper_bounds(xmin, xmax, T[d, :], item_features[:, k])
                user_features[d, k] = find_element(item_features[:, k], train[d, :], M[d, :], T[d, :], l.max(), u.min())
            for n in range(N):
                l = lower_bounds(xmin, xmax, T[:, n], user_features[:, k])
                u = upper_bounds(xmin, xmax, T[:, n], user_features[:, k])
                item_features[n, k] = find_element(user_features[:, k], train[:, n], M[:, n], T[:, n], l.max(), u.min())

        # train error
        P = pred_fn(user_features, item_features, None, None)
        err = err_fn(train, P, O)

        # loss value
        loss = 0.5 * (np.sum(np.power(M * (train - P), 2))/num_nz)

        # store best model
        if loss < min_l:
            min_l = loss
            best_W = user_features.copy()
            best_Z = item_features.copy()

        if display and not e % display:
            print("epoch: {:4d}, loss: {:e} -- {} on training set: {:e} .".format(e, loss, err_fn.__name__, err))
        loss_list.append(loss)
        err_list.append(err)
        change = np.fabs(loss_list[-1] - loss_list[-2]) / np.fabs(loss_list[-1])
        e += 1

    # restore best model
    user_features = best_W
    item_features = best_Z

    # best model's loss & train error
    P = pred_fn(user_features, item_features, None, None)
    loss = 0.5 * (np.sum(np.power(O * (train - P), 2)) / num_nz)
    err = err_fn(train, P, O)
    loss_list.append(loss)
    err_list.append(err)
    print("epoch: {:4d}, loss: {:e} -- {} on training set: {:e} .".format(e, loss, err_fn.__name__, err))

    return user_features, item_features, None, None, loss_list[1:], err_list, pred_fn
