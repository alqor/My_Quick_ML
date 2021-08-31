import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error

def find_cat_columns(df):
    """base on np.dtype.kind
    O - object
    S - (byte-)string
    U - Unicode
    """
    return [col for col in df.columns if df[col].dtype.kind in 'OSU']


def find_num_columns(df):
    """base on np.dtype.kind
    b - boolean
    i - signed integer
    u - unsigned integer
    f - floating-point
    c - complex floating-point
    """
    return [col for col in df.columns if df[col].dtype.kind in 'biufc']



def get_ohe(df, columns):
    """Create dummies for specified columns in dataframe"""
    mi = 0
    for cat_col in columns:
        one_dummy_df = pd.get_dummies(df[cat_col], prefix=cat_col)
        if mi == 0:
            ohe_df = one_dummy_df
        else:
            ohe_df = ohe_df.merge(one_dummy_df, right_index=True, left_index=True)
        mi+=1
    ohe_df.index = df.index
    ohe_df['index'] = df.index
    return ohe_df


def get_ordinal(df, columns, mode='train'):
    """create df with ordinal coded columns
    use 'train' mode for train set
    and 'test' or 'valid' for validation or test datasets
    """
    ordinal_encoder = OrdinalEncoder()
    new_columns = ['ord_'+col for col in columns]
    ord_df = pd.DataFrame(columns=columns)
    
    ord_df['index'] = df.index
    ord_df.index=df.index
    
    if mode=='train':
        ord_df[columns] = ordinal_encoder.fit_transform(df[columns])
    if mode=='test' or mode=='valid':
        ord_df[columns] = ordinal_encoder.fit_transform(df[columns])
        ord_df[columns] = ordinal_encoder.transform(df[columns])
    
    ord_df['index'] = df.index
    ord_df.index=df.index
    ord_df.rename(columns=dict(zip(columns, new_columns)), inplace=True)
    return ord_df


def get_frequencies(df, columns):
    """Create df with frequency-encoded columns"""
    freq_df = pd.DataFrame()
    
    freq_df['index'] = df.index
    freq_df.index=df.index
    for col in columns:
        counts = dict(df[col].value_counts() / len(df))
        freq_df[col+'_freq'] = df[col].replace(counts)
    
    return freq_df


def get_squars(df, columns):
    sq_df = pd.DataFrame()
    sq_df['index'] = df.index
    sq_df.index=df.index

    for col in columns:
        sq_df['sq_'+col] =df[col]**2
    
    return sq_df


def basic_preproc_pipe(X_df, columns_to_drop=[], 
                             ohe_columns=[],
                             ord_columns=[],
                             freq_columns=[],
                             sq_columns=[],
                             ord_mode='train', 
                             **kwargs):
    """Preprocess data for each fold"""

    new_X = X_df.copy()
    new_X['index'] = X_df.index
    
    if ohe_columns:
        ohe_df = get_ohe(X_df, ohe_columns)
        new_X = new_X.merge(ohe_df, on='index', how='left')
    
    if ord_columns:
        ord_df = get_ordinal(X_df, ord_columns, mode=ord_mode)
        new_X = new_X.merge(ord_df, on='index', how='left')

    if freq_columns:
        freq_df = get_frequencies(X_df, freq_columns)
        new_X = new_X.merge(freq_df, on='index', how='left')

    if sq_columns:
        sq_df = get_squars(X_df, sq_columns)
        new_X = new_X.merge(sq_df, on='index', how='left')

    if columns_to_drop:
        new_X.drop(columns_to_drop, axis=1, inplace=True)

    new_X.index = X_df.index
    new_X.drop('index', axis=1, inplace=True)

    return new_X


def create_folds(df, n_folds):
    """Create n-folds number of folds as separate column in df"""

    X_folds = df.copy()
    X_folds.reset_index(inplace=True)
    X_folds['kfold'] = -1
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (train_indicies, valid_indicies) in enumerate(kf.split(X_folds)):
        X_folds.loc[valid_indicies, 'kfold'] = fold
    
    X_folds.index = X_folds['id']
    X_folds.drop('id', axis=1, inplace=True)

    return X_folds


def splitter_by_fold(X_folds_df, fold, unuseful_features):
    """split df by folds"""

    useful_features = [col for col in X_folds_df.columns if col not in unuseful_features]

    X_train = X_folds_df[X_folds_df['kfold'] != fold]
    X_valid = X_folds_df[X_folds_df['kfold'] == fold]

    y_train = X_train['target']
    y_valid = X_valid['target']

    X_train = X_train[useful_features]
    X_valid = X_valid[useful_features]

    return X_train, y_train, X_valid, y_valid

def prediction_collector(model, fold, 
                         X_train, X_test, X_valid, y_train, y_valid,
                         test_predictions,
                         valid_predictions,
                         scores):

    preds_valid = model.predict(X_valid)
    valid_ids = list(X_valid.index)

    valid_predictions.update(dict(zip(valid_ids, preds_valid)))

    preds_test = model.predict(X_test)
    test_predictions.append(preds_test)

    rmse = mean_squared_error(y_valid, preds_valid, squared=False)

    print(fold, rmse)
    scores.append(rmse)



def fit_predict_with_folds_rmse(X_folds_df, 
                                test,
                                n_folds, 
                                unuseful_features,
                                columns_to_drop, 
                                ohe_columns,
                                ord_columns,
                                freq_columns,
                                sq_columns,
                                model_algorithm, 
                                train_params, 
                                fit_params,
                                files_prefix):
    """fit model with specified parameters on X_folds df"""

    test_predictions=[]
    valid_predictions={}
    scores=[]

    for fold in range(n_folds):
        # splitting
        X_train, y_train, X_valid, y_valid = splitter_by_fold(X_folds_df, fold, unuseful_features)
        X_test = test.copy()

        # preprocessing
        X_train = basic_preproc_pipe(X_train,  
                                     columns_to_drop, 
                                     ohe_columns,
                                     ord_columns,
                                     freq_columns,
                                     sq_columns, 
                                     ord_mode='train')

        X_valid = basic_preproc_pipe(X_valid, 
                                     columns_to_drop, 
                                     ohe_columns,
                                     ord_columns,
                                     freq_columns,
                                     sq_columns,  
                                     ord_mode='valid')

        X_test = basic_preproc_pipe(X_test, 
                                     columns_to_drop, 
                                     ohe_columns,
                                     ord_columns,
                                     freq_columns,
                                     sq_columns, 
                                     ord_mode='test')
        
        evaluation = [( X_valid, y_valid)]
        fit_params['eval_set'] = evaluation

        model = model_algorithm(**train_params)
        model.fit(X_train, y_train,
                    **fit_params)

        prediction_collector(model, fold,
                         X_train, X_test, X_valid, y_train, y_valid,
                         test_predictions=test_predictions,
                         valid_predictions=valid_predictions,
                         scores=scores
                        )

    print(np.mean(scores), np.std(scores))

    #save both valid and test predictions
    valid_predictions_df = pd.DataFrame.from_dict(data=valid_predictions, orient='index').reset_index()
    valid_predictions_df.columns = ['id', 'prediction']
    valid_predictions_df.sort_values(by='id', inplace=True)
    valid_predictions_df.to_csv(f'predictions/{files_prefix}_valid_predictions.csv', index=False)

    test_predictions_df = pd.DataFrame({'id': test.index, 
                                        'target': np.mean(np.column_stack(test_predictions),
                                            axis=1)})
    test_predictions_df.to_csv(f'predictions/{files_prefix}_test_predictions.csv', index=False)

    return valid_predictions_df, test_predictions_df, scores


def create_train_test_stack(files=[], mode='test', original_train=None):
    i=1
    for f in files:
        df = pd.read_csv(f)
        if mode=='test':
            model_name = f.split('_test_predictions.csv')[0].split('/')[1]
            df.rename(columns={'target':'prediction_'+model_name}, inplace=True)
        if mode=='train':
            model_name = f.split('_valid_predictions.csv')[0].split('/')[1]
            df.rename(columns={'prediction':'prediction_'+model_name}, inplace=True)
        if i==1:
            res = df
        else:
            res = res.merge(df, on='id', how='left')
        i+=1
    res.index = res['id']
    res.drop('id', axis=1, inplace=True)
    if mode=='train':
        res['target'] = original_train['target']
    return res