import pandas as pd


xgb_preds = pd.read_csv('xgb_preds.csv');
sgd_fm = pd.read_csv('../kaggle_quora_benchmark/sgd_fm.csv')

sub = pd.DataFrame()
sub['test_id'] = xgb_preds['test_id']
sub['is_duplicate'] = xgb_preds['is_duplicate'] * 0.95 + sgd_fm['is_duplicate'] * 0.05
sub.to_csv('ensemble_preds.csv', index=False)
