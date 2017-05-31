# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


MISSING = -99999


def load_data(file_name):
    data = pd.read_csv('../data/%s' % file_name, encoding='utf-8')

    return data


def get_train_validation_sets(testset=False):
    data = load_data('test_features_v3.csv' if testset else 'train_features_v3.csv')
    data.fillna(MISSING, inplace=True)

    if testset:
        print 'Prepare test set'
        test_id = data['test_id']
        x_test = data.drop(['test_id', 'question1', 'question2'], axis=1)

        return x_test, test_id
    else:
        print 'Prepare train and validation sets'
        y_train = data['is_duplicate']
        x_train = data.drop(['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'], axis=1)

        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

        return x_train, x_valid, y_train, y_valid


if __name__ == '__main__':
    submission_mode = False
    x_train, x_valid, y_train, y_valid = get_train_validation_sets()

    # Set our parameters for xgboost
    if submission_mode:
        class_weights = {0: 1 - (0.165 / (1 - 0.165)), 1: 0.165 / (1 - 0.165)}
    else:
        class_weights = {0: 1 - 0.165, 1: 0.165}
    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.05  # 0.0125 - 2400 iter, 0.05 - 600 iter
    params['max_depth'] = 16  # 11
    params['subsample'] = 0.8
    params['max_delta_step'] = 1
    params['min_child_weight'] = 2
    if submission_mode:
        params['scale_pos_weight'] = 0.165 / (1 - 0.165)

    d_train = xgb.DMatrix(x_train, label=y_train, missing=MISSING)
    d_valid = xgb.DMatrix(x_valid, label=y_valid, missing=MISSING)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    print 'Start training'

    bst = xgb.train(params, d_train, 600, watchlist, early_stopping_rounds=50, verbose_eval=10)  # 2400
    xgb_preds_valid = bst.predict(d_valid)
    val = pd.DataFrame()
    val['test_id'] = range(x_valid.shape[0])
    val['is_duplicate'] = xgb_preds_valid
    val.to_csv('xgb_valid.csv', index=False)
    del d_train, d_valid, x_train, x_valid, val, xgb_preds_valid

    if not submission_mode:
        exit(0)

    x_test, test_id = get_train_validation_sets(testset=True)

    xgb_preds = bst.predict(xgb.DMatrix(x_test, missing=MISSING))

    print("Writing output...")
    sub = pd.DataFrame()
    sub['test_id'] = test_id
    sub['is_duplicate'] = xgb_preds
    sub.to_csv('xgb_preds.csv', index=False)
