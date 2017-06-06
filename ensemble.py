import pandas as pd

xgb_preds = pd.read_csv('xgb_preds.csv')
xgb_over = pd.read_csv('xgb_preds_oversample.csv')
sgd_fm = pd.read_csv('../kaggle_quora_benchmark/sgd_fm.csv')

sub = pd.DataFrame()
sub['test_id'] = xgb_preds['test_id']
sub['is_duplicate'] = xgb_preds['is_duplicate'] * 0.7 + xgb_over['is_duplicate'] * 0.15 + sgd_fm['is_duplicate'] * 0.15
sub.to_csv('ensemble_preds.csv', index=False)
