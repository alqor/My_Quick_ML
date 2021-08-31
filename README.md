# My_Quick_ML

1. Some quick notes for 30DaysOfML Kaggle competition
2. Could be used to similar data sets (tabular data with mixed categorical and numeric columns)

# How to use
1. With main file in the root you could train 3 models: XGBoost, LGBM and Catboost
2. Parameters could be rewrite manually in parameters.py
3. In settings total number of folds for train/valid splits could be set as well as columns for specific preprocessing

(more detailed description will be added latter)

# Known Issues
1. Tunning hyperparameters with Optuna (hyperparams_tunning folder) is slightly bugged and should be rewrite 
(also it should be one .ipynb file to run it simply in side notebooks like KAGGLE, because parameter tunning takes a lot of time and it's better to run such tasks in clouds then in your workingg machine)
2. Specified folders to store predictions and train data didn't move to github repo because of .gitignore (it bans all .csv) = try to add __init__.py (??) or look for better solution

# To Be Add
1. Some basic notebbok with EDA
2. Feature selection (Boruta-SHAP?) or something else..
