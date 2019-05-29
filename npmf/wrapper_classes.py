# -*- coding: utf-8 -*-

"""
Created by:         Emanuele Bugliarello (@e-bug)
Date created:       7/17/2018
Date last modified: 5/24/2019
"""

from __future__ import absolute_import, division, print_function

from npmf.error_metrics import *
from npmf.init_functions import *


# ==================================================================================================================== #
#                                                                                                                      #
#                                             Matrix Factorization Classes                                             #
#                                                                                                                      #
# ==================================================================================================================== #

class MF(object):
    def __init__(self, algorithm, init_fn=rand_init, num_features=6, nanvalue=0, xmin=None, xmax=None,
                 lr0=None, decay_fn=None, batch_size=None,
                 lambda_user=0.1, lambda_item=0.1, max_epochs=2000, int_iter=None, stop_criterion=1e-6,
                 err_fn=rmse, display=1, seed=42):
        self.algorithm = algorithm
        self.init_fn = init_fn
        self.num_features = num_features
        self.xmin = xmin
        self.xmax = xmax
        self.lr0 = lr0
        self.decay_fn = decay_fn
        self.batch_size = batch_size
        self.lambda_user = lambda_user
        self.lambda_item = lambda_item
        self.max_epochs = max_epochs
        self.int_iter = int_iter
        self.stop_criterion = stop_criterion
        self.err_fn = err_fn
        self.display = display
        self.nanvalue = nanvalue
        self.seed = seed

        self.user_features = None
        self.item_features = None
        self.user_biases = None
        self.item_biases = None
        self.pred_fn = None

        self.loss_list = []
        self.err_list = []
        self.train_errors = dict()
        self.valid_errors = dict()
        self.test_errors = dict()

    def fit(self, matrix, **kwargs):
        self.user_features,self.item_features,self.user_biases,self.item_biases,self.loss_list,self.err_list,self.pred_fn = \
            self.algorithm(train=matrix, init_fn=self.init_fn, num_features=self.num_features, nanvalue=self.nanvalue, 
                           xmin=self.xmin, xmax=self.xmax,
                           lr0=self.lr0, decay_fn=self.decay_fn, batch_size=self.batch_size,
                           lambda_user=self.lambda_user, lambda_item=self.lambda_item,
                           max_epochs=self.max_epochs, int_iter=self.int_iter,
                           stop_criterion=self.stop_criterion, err_fn=self.err_fn, display=self.display, seed=self.seed)
        self.train_errors[self.err_fn.__name__] = self.err_list[-1]

    def predict(self):
        if self.user_features is None or self.item_features is None:
            raise RuntimeError('Model not fit yet')
        return self.pred_fn(self.user_features, self.item_features, self.user_biases, self.item_biases)

    def score(self, err_fn, matrix, err_type):
        if self.user_features is None or self.item_features is None:
            raise RuntimeError('Model not fit yet')
        O = matrix != self.nanvalue
        if self.xmin is not None and self.xmax is not None:
            matrix = (matrix - self.xmin) / (self.xmax - self.xmin)
        P = self.pred_fn(self.user_features, self.item_features, self.user_biases, self.item_biases)
        err = err_fn(matrix, P, O)
        err_type = err_type.lower()
        if err_type == 'train':
            self.train_errors[err_fn.__name__] = err
        elif err_type == 'validation':
            self.valid_errors[err_fn.__name__] = err
        elif err_type == 'test':
            self.test_errors[err_fn.__name__] = err
        print("{} on {} set: {:e} .".format(err_fn.__name__, err_type, err))
        del O, P
        return err


class WeightedMF(MF):
    def fit(self, matrix, **kwargs):
        confidence = kwargs['confidence']
        self.user_features,self.item_features,self.user_biases,self.item_biases,self.loss_list,self.err_list,self.pred_fn = \
            self.algorithm(train=matrix, init_fn=self.init_fn, num_features=self.num_features,
                           nanvalue=self.nanvalue,
                           confidence=confidence,
                           xmin=self.xmin, xmax=self.xmax,
                           lr0=self.lr0, decay_fn=self.decay_fn, batch_size=self.batch_size,
                           lambda_user=self.lambda_user, lambda_item=self.lambda_item,
                           max_epochs=self.max_epochs, int_iter=self.int_iter,
                           stop_criterion=self.stop_criterion, err_fn=self.err_fn, display=self.display,
                           seed=self.seed)
        self.train_errors[self.err_fn.__name__] = self.err_list[-1]


# ============================================= Cross-validation classes ============================================= #

class CvMF(object):
    def __init__(self, algorithm, init_fn=rand_init, num_features=6, nanvalue=0, xmin=None, xmax=None,
                 lr0=None, decay_fn=None, batch_size=None,
                 lambda_user=0.1, lambda_item=0.1, max_epochs=2000, int_iter=None, stop_criterion=1e-6,
                 err_fn=rmse, display=1, seed=42):
        self.algorithm = algorithm
        self.init_fn = init_fn
        self.num_features = num_features
        self.xmin = xmin
        self.xmax = xmax
        self.lr0 = lr0
        self.decay_fn = decay_fn
        self.batch_size = batch_size
        self.lambda_user = lambda_user
        self.lambda_item = lambda_item
        self.max_epochs = max_epochs
        self.int_iter = int_iter
        self.stop_criterion = stop_criterion
        self.err_fn = err_fn
        self.display = display
        self.nanvalue = nanvalue
        self.seed = seed

        self.user_features_list = []
        self.item_features_list = []
        self.user_biases_list = []
        self.item_biases_list = []
        self.pred_fn = None

        self.loss_lists_list = []
        self.err_lists_list = []
        self.train_errors_list = []
        self.valid_errors_list = []
        self.test_errors_list = []

        self.train_error_agg = dict()
        self.train_error_dev = dict()
        self.valid_error_agg = dict()
        self.valid_error_dev = dict()
        self.test_error_agg = dict()
        self.test_error_dev = dict()

    def fit(self, matrices_list, **kwargs):
        if len(self.train_errors_list) != len(matrices_list):
            self.train_errors_list = [dict() for _ in range(len(matrices_list))]
            self.valid_errors_list = [dict() for _ in range(len(matrices_list))]
            self.test_errors_list = [dict() for _ in range(len(matrices_list))]
        for i, matrix in enumerate(matrices_list):
            user_features, item_features, user_biases, item_biases, loss_list, err_list, self.pred_fn = \
                self.algorithm(train=matrix, init_fn=self.init_fn, num_features=self.num_features, 
                               nanvalue=self.nanvalue, xmin=self.xmin, xmax=self.xmax,
                               lr0=self.lr0, decay_fn=self.decay_fn, batch_size=self.batch_size,
                               lambda_user=self.lambda_user, lambda_item=self.lambda_item, 
                               max_epochs=self.max_epochs, int_iter=self.int_iter,
                               stop_criterion=self.stop_criterion, err_fn=self.err_fn, 
                               display=self.display, seed=self.seed)
            self.user_features_list.append(user_features)
            self.item_features_list.append(item_features)
            self.user_biases_list.append(user_biases)
            self.item_biases_list.append(item_biases)
            self.loss_lists_list.append(loss_list)
            self.err_lists_list.append(err_list)
            self.train_errors_list[i][self.err_fn.__name__] = err_list[-1]

    def predict(self):
        if len(self.user_features_list) == 0 or len(self.item_features_list) == 0:
            raise RuntimeError('Models not fit yet')
        matrices_list = []
        for i in range(len(self.user_features_list)):
            matrices_list.append(self.pred_fn(self.user_features_list[i], self.item_features_list[i],
                                              self.user_biases_list[i], self.item_biases_list[i]))
        return matrices_list

    def score(self, err_fn, matrices_list, err_type, agg_fn, dev_fn):
        if len(self.user_features_list) == 0 or len(self.item_features_list) == 0:
            raise RuntimeError('Models not fit yet')
        errs = []
        for i, matrix in enumerate(matrices_list):
            O = matrix != self.nanvalue
            if self.xmin is not None and self.xmax is not None:
                matrix = (matrix - self.xmin) / (self.xmax - self.xmin)
            P = self.pred_fn(self.user_features_list[i], self.item_features_list[i],
                             self.user_biases_list[i], self.item_biases_list[i])
            errs.append(err_fn(matrix, P, O))
        err_agg = agg_fn(errs)
        err_dev = dev_fn(errs)
        err_type = err_type.lower()
        if err_type == 'train':
            self.train_errors_list = [dict()] * len(errs)
            for i in range(len(errs)):
                self.train_errors_list[i][err_fn.__name__] = errs[i]
            self.train_error_agg[err_fn.__name__ + '_' + agg_fn.__name__] = err_agg
            self.train_error_dev[err_fn.__name__ + '_' + agg_fn.__name__] = err_dev
        elif err_type == 'validation':
            self.valid_errors_list = [dict()] * len(errs)
            for i in range(len(errs)):
                self.valid_errors_list[i][err_fn.__name__] = errs[i]
            self.valid_error_agg[err_fn.__name__ + '_' + agg_fn.__name__] = err_agg
            self.valid_error_dev[err_fn.__name__ + '_' + agg_fn.__name__] = err_dev
        elif err_type == 'test':
            self.test_errors_list = [dict()] * len(errs)
            for i in range(len(errs)):
                self.test_errors_list[i][err_fn.__name__] = errs[i]
            self.test_error_agg[err_fn.__name__ + '_' + agg_fn.__name__] = err_agg
            self.test_error_dev[err_fn.__name__ + '_' + agg_fn.__name__] = err_dev
        print("{} {} on {} data: {:e}, {}:  {:e}.".format(agg_fn.__name__, err_fn.__name__, err_type,
                                                          err_agg, dev_fn.__name__, err_dev))
        del O, P
        return err_agg, err_dev


class CvWeightedMF(CvMF):
    def fit(self, matrices_list, **kwargs):
        confidence = kwargs['confidence']
        if len(self.train_errors_list) != len(matrices_list):
            self.train_errors_list = [dict() for _ in range(len(matrices_list))]
            self.valid_errors_list = [dict() for _ in range(len(matrices_list))]
            self.test_errors_list = [dict() for _ in range(len(matrices_list))]
        for i, matrix in enumerate(matrices_list):
            user_features, item_features, user_biases, item_biases, loss_list, err_list, self.pred_fn = \
                self.algorithm(train=matrix, init_fn=self.init_fn, num_features=self.num_features,
                               nanvalue=self.nanvalue, xmin=self.xmin, xmax=self.xmax, confidence=confidence,
                               lr0=self.lr0, decay_fn=self.decay_fn, batch_size=self.batch_size,
                               lambda_user=self.lambda_user, lambda_item=self.lambda_item,
                               max_epochs=self.max_epochs, int_iter=self.int_iter,
                               stop_criterion=self.stop_criterion, err_fn=self.err_fn,
                               display=self.display, seed=self.seed)
            self.user_features_list.append(user_features)
            self.item_features_list.append(item_features)
            self.user_biases_list.append(user_biases)
            self.item_biases_list.append(item_biases)
            self.loss_lists_list.append(loss_list)
            self.err_lists_list.append(err_list)
            self.train_errors_list[i][self.err_fn.__name__] = err_list[-1]

