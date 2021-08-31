"""
Erly defined best parameters;
or test parameters.

Should be generate automatic, but to save current 
"""

### XGBOOST
xgboost_base_train_parameters = {
    'n_estimators': 7000,
    'random_state': 42,
}

xgboost_train_parameters = {
    'learning_rate': 0.03136358498374816,
    'reg_lambda': 0.06607375853720447,
    'reg_alpha': 20.51028241497515,
    'subsample': 0.9464704876460253,
    'colsample_bytree': 0.1338118715164083,
    'max_depth': 3}

xgboost_train_parameters.update(xgboost_base_train_parameters)

xgboost_fit_parameters = {
    'eval_metric': "rmse",
    'early_stopping_rounds': 20,
    'verbose': False
}

### LGBM

lgbm_base_train_parameters = {
    'boosting_type': 'gbdt',
    'metric': 'rmse',
    'n_jobs': -1,
    'verbose': -1,
    'random_state': 42,
    'objective': 'poisson',
    'reg_sqrt': False,
    'n_estimators': 7000, 
}

lgbm_train_parameters = {
    'learning_rate': 0.19814465389718905,
    'reg_lambda': 7.755091061719783e-08,
    'reg_alpha': 16.673360279155897,
    'subsample': 0.9170902286921229,
    'colsample_bytree': 0.10107320101105535,
    'max_depth': 2,
    'num_leaves': 159}

lgbm_train_parameters.update(lgbm_base_train_parameters)

lgbm_fit_parameters = {
    'eval_metric': 'rmse',
    'early_stopping_rounds': 100,
    'verbose': -1
}


### CAT

cat_base_train_parameters = {
    'random_state': 2,
    'grow_policy': 'Depthwise',
    'iterations': 10000,  # 7000
    'use_best_model': True,
    'eval_metric': 'RMSE',
    'od_type': 'iter',
    'od_wait': 20,
    'logging_level': 'Silent',
}

cat_train_parameters = {
    'learning_rate': 0.096769348223102,
    'l2_leaf_reg': 1.5,
    'min_child_samples': 4,
    'depth': 2}

cat_train_parameters.update(cat_base_train_parameters)

cat_fit_parameters = {
    'early_stopping_rounds': 100,
    'verbose': False,
}


stack_params = {
    'n_estimators': 10000,
    'random_state': 42,

    'learning_rate': 0.01191446089739685,
    'reg_lambda': 4.3804012718388296e-05,
    'reg_alpha': 3.2463091972013394,
    'subsample': 0.1508962458061545,
    'colsample_bytree': 0.9026436740386892,
    'max_depth': 3
}
