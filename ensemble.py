import pandas as pd
from sklearn.metrics import log_loss

from xgb import get_train_validation_sets


#_, _, _, y_val = get_train_validation_sets()
#xgb_val = pd.read_csv('xgb_valid.csv');
#sgd_fm_val = pd.read_csv('../kaggle_quora_benchmark/valid.csv')
#print log_loss(y_val, xgb_val['is_duplicate'].values * 0.95 + sgd_fm_val['is_duplicate'].values * 0.05)

xgb_preds = pd.read_csv('xgb_preds.csv')
sgd_fm = pd.read_csv('../kaggle_quora_benchmark/sgd_fm.csv')

sub = pd.DataFrame()
sub['test_id'] = xgb_preds['test_id']
sub['is_duplicate'] = xgb_preds['is_duplicate'] * 0.9 + sgd_fm['is_duplicate'] * 0.1
sub.to_csv('ensemble_preds.csv', index=False)
