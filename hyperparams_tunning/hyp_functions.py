import optuna
from pathlib import Path
import sys
import os
import json
import numpy as np

PAR_DIR = Path(__file__).parents[1]
sys.path.append(str(PAR_DIR))

from functions import *
from settings import *
from parameters import *

def get_best_params(direction,
                    n_trilas,
                    X_folds,
                    fold,
                    param_dict,
                    unuseful_features,
                    columns_to_drop,
                    ohe_columns,
                    ord_columns,
                    freq_columns,
                    sq_columns,
                    model_algorithm,
                    base_train_params,
                    fit_params,
                    file_to_write=None):

    def objective(trial):
        trials_dict = {
            'LGBM': {
                'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.4, log=True),
                'reg_lambda': trial.suggest_loguniform("reg_lambda", 1e-8, 100.0),
                'reg_alpha': trial.suggest_loguniform("reg_alpha", 1e-8, 100.0),
                'subsample': trial.suggest_float("subsample", 0.1, 1.0),
                'colsample_bytree': trial.suggest_float("colsample_bytree", 0.1, 1.0),
                'max_depth': trial.suggest_int("max_depth", 1, 10),
                'num_leaves': trial.suggest_int('num_leaves', 10, 256),
            },

            'XGBOOST': {
                'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                'reg_lambda': trial.suggest_loguniform("reg_lambda", 1e-8, 100.0),
                'reg_alpha': trial.suggest_loguniform("reg_alpha", 1, 100.0),
                'subsample': trial.suggest_float("subsample", 0.1, 1.0),
                'colsample_bytree': trial.suggest_float("colsample_bytree", 0.1, 1.0),
                'max_depth': trial.suggest_int("max_depth", 1, 7),
            },

            'CAT': {
                'learning_rate': trial.suggest_float("learning_rate", 0.05, 0.25, log=True),
                'l2_leaf_reg': trial.suggest_discrete_uniform('l2_leaf_reg', 1.0, 5.5, 0.5),
                'min_child_samples': trial.suggest_categorical('min_child_samples', [1, 4, 8, 16, 32]),
                'depth': trial.suggest_int("depth", 1, 15),
            }
        }

        base_train_params.update(trials_dict[param_dict])

        X_train, y_train, X_valid, y_valid = splitter_by_fold(
            X_folds, fold, unuseful_features)
        
        if param_dict == 'CAT':
            cat_fit_parameters['cat_features'] = np.where(X_folds.dtypes == object)[0]

        X_train = basic_preproc_pipe(X_train,
                                     columns_to_drop=columns_to_drop,
                                     ohe_columns=ord_columns,
                                     ord_columns=ord_columns,
                                     freq_columns=freq_columns,
                                     sq_columns=sq_columns,
                                     ord_mode='train',
                                     )

        X_valid = basic_preproc_pipe(X_valid,
                                     columns_to_drop=columns_to_drop,
                                     ohe_columns=ord_columns,
                                     ord_columns=ord_columns,
                                     freq_columns=freq_columns,
                                     sq_columns=sq_columns,
                                     ord_mode='valid',
                                     )

        evaluation = [(X_valid, y_valid)]
        fit_params['eval_set'] = evaluation

        model = model_algorithm(**base_train_params)
        model.fit(X_train, y_train,
                  **fit_params)
        preds_valid = model.predict(X_valid)
        rmse = mean_squared_error(y_valid, preds_valid, squared=False)
        return rmse

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trilas)

    print(f'Best Params: {study.best_params}')
    if file_to_write:
        with open(file_to_write+'.txt', 'w') as file:
            file.write(json.dumps(study.best_params))
    return study.best_params
