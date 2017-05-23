# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


MISSING = -99999


def load_data(file_name):
    data = pd.read_csv('../data/%s' % file_name, encoding='utf-8')

    return data


def get_train_validation_sets(testset=False, reverse=False):
    data = load_data('test_features_v2.csv' if testset else 'train_features_v2.csv')
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

        if reverse:
            train_reverse = load_data('train_reverse_features_v2.csv')
            train_reverse.fillna(MISSING, inplace=True)
            y_train_reverse = train_reverse['is_duplicate']
            x_train_reverse = train_reverse.drop(['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'], axis=1)

            x_train, x_valid, y_train, y_valid, x_train_reverse, x_valid_reverse,\
            y_train_reverse, y_valid_reverse = train_test_split(
                x_train, y_train, x_train_reverse, y_train_reverse, test_size=0.2, random_state=4242)

            return x_train, x_valid, y_train, y_valid, x_valid_reverse

            x_train = pd.concat([x_train, x_train_reverse])
            y_train = pd.concat([y_train, y_train_reverse])
        else:
            x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

        return x_train, x_valid, y_train, y_valid


if __name__ == '__main__':
    submission_mode = True
    x_train, x_valid, y_train, y_valid = get_train_validation_sets(reverse=False)
    #x_train, x_valid, y_train, y_valid, x_valid_reverse = get_train_validation_sets(reverse=True)

    # Set our parameters for xgboost
    if submission_mode:
        class_weights = {0: 1 - (0.165 / (1 - 0.165)), 1: 0.165 / (1 - 0.165)}
    else:
        class_weights = {0: 1 - 0.165, 1: 0.165}
    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.0125  # 0.0125 - 2400 iter, 0.05 - 600 iter
    params['max_depth'] = 16  # 11
    params['subsample'] = 0.8
    params['max_delta_step'] = 1
    params['min_child_weight'] = 2
    if submission_mode:
        params['scale_pos_weight'] = 0.165 / (1 - 0.165)

    d_train = xgb.DMatrix(x_train, label=y_train, missing=MISSING)
    d_valid = xgb.DMatrix(x_valid, label=y_valid, missing=MISSING)
    #d_valid_reverse = xgb.DMatrix(x_valid_reverse, label=y_valid, missing=MISSING)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    print 'Start training'

    bst = xgb.train(params, d_train, 2400, watchlist, early_stopping_rounds=50, verbose_eval=10)  # 2400
    del d_train, d_valid

    clf = ExtraTreesClassifier(n_estimators=600, criterion='entropy', random_state=999,
                               class_weight=class_weights)
    x_train2 = x_train.replace([np.inf, -np.inf], np.nan).fillna(MISSING)
    x_valid2 = x_valid.replace([np.inf, -np.inf], np.nan).fillna(MISSING)
    clf.fit(x_train2, y_train)
    et_preds = clf.predict_proba(x_valid2)
    print 'Extra Trees:', log_loss(y_valid, et_preds)

    xgb_preds = bst.predict(xgb.DMatrix(x_valid.fillna(MISSING), missing=MISSING))
    print 'XGB and Extra Trees:', log_loss(y_valid, (0.1 * et_preds[:, 1] + 0.9 * xgb_preds))

    #import pdb
    #pdb.set_trace()

    del x_train, x_valid, x_train2, x_valid2, et_preds, xgb_preds

    x_test, test_id = get_train_validation_sets(testset=True)
    x_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    x_test.fillna(MISSING, inplace=True)

    et_preds = clf.predict_proba(x_test)
    del clf
    xgb_preds = bst.predict(xgb.DMatrix(x_test, missing=MISSING))
    preds = (0.1 * et_preds[:, 1] + 0.9 * xgb_preds)  # Ensemble XGBoost and ExtraTrees

    print("Writing output...")
    sub = pd.DataFrame()
    sub['test_id'] = test_id
    sub['is_duplicate'] = preds
    sub.to_csv('xgb_preds.csv', index=False)

    import pdb
    pdb.set_trace()
