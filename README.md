# Kaggle: Quora question pairs
Solution from competition

Run:
```
python feature_engineering.py
python xgb.py (with positive class weight parameter passed to xgb)
python xgb.py (with oversampling)
<build sgd_fm from https://github.com/qqgeogor/kaggle_quora_benchmark>
python ensemble.py
```
