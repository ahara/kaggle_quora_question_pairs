import pandas as pd

xgb_preds = pd.read_csv('xgb_preds.csv')
xgb_leaky = pd.read_csv('XGB_leaky.csv')
sgd_fm = pd.read_csv('../kaggle_quora_benchmark/sgd_fm.csv')

sub = pd.DataFrame()
sub['test_id'] = xgb_preds['test_id']
sub['is_duplicate'] = xgb_preds['is_duplicate'] * 0.5 + xgb_leaky['is_duplicate'] * 0.4 + sgd_fm['is_duplicate'] * 0.1
sub.to_csv('ensemble_preds.csv', index=False)
