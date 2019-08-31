"""
Tree Boosting Classifier using multiple boosting techniques and stacking their results by taking
a weighted average of each models' predictions.

Used to solve Kaggle Santander Customer Transaction problem.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import pickle
import os
import gc
import time

gc.enable()


''' CONFIGURABLE PARAMETERS '''
NUM_FOLDS = 5

TRAIN_LGB = False
TRAIN_XGB = True
TRAIN_CB = False

LGB_USE_GPU = True
XGB_USE_GPU = True
CB_USE_GPU = True

MODELS_PATH = './models/'
SUBMISSIONS_PATH = './submissions/'
DATASET_PATH = './dataset/'
''' CONFIGURABLE PARAMETERS '''


def fit_lgb(X_fit, y_fit, X_val, y_val, counter, lgb_path, name):
    model = lgb.LGBMClassifier(max_depth=-1,
                               n_estimators=999999,
                               learning_rate=0.02,
                               colsample_bytree=0.3,
                               num_leaves=2,
                               metric='auc',
                               objective='binary',
                               n_jobs=-1,
                               device=("gpu" if LGB_USE_GPU else "cpu"))

    model.fit(X_fit, y_fit,
              eval_set=[(X_val, y_val)],
              verbose=0,
              early_stopping_rounds=1000)

    cv_val = model.predict_proba(X_val)[:, 1]

    # Save LightGBM Model
    save_to = '{}{}_fold{}.txt'.format(lgb_path, name, counter + 1)
    model.booster_.save_model(save_to)

    return cv_val


def fit_xgb(X_fit, y_fit, X_val, y_val, counter, xgb_path, name):
    model = xgb.XGBClassifier(max_depth=2,
                              n_estimators=999999,
                              colsample_bytree=0.3,
                              learning_rate=0.02,
                              objective='binary:logistic',
                              n_jobs=-1,
                              gpu_hist=XGB_USE_GPU)

    model.fit(X_fit, y_fit,
              eval_set=[(X_val, y_val)],
              verbose=False,
              early_stopping_rounds=1000)

    cv_val = model.predict_proba(X_val)[:, 1]

    # Save XGBoost Model
    save_to = '{}{}_fold{}.dat'.format(xgb_path, name, counter + 1)
    pickle.dump(model, open(save_to, "wb"))

    return cv_val


def fit_cb(X_fit, y_fit, X_val, y_val, counter, cb_path, name):
    model = cb.CatBoostClassifier(iterations=999999,
                                  max_depth=2,
                                  learning_rate=0.02,
                                  colsample_bylevel=0.03,
                                  objective="Logloss",
                                  task_type=("GPU" if CB_USE_GPU else "CPU"))

    model.fit(X_fit, y_fit,
              eval_set=[(X_val, y_val)],
              verbose=0, early_stopping_rounds=1000)

    cv_val = model.predict_proba(X_val)[:, 1]

    # Save Catboost Model
    save_to = "{}{}_fold{}.mlmodel".format(cb_path, name, counter + 1)
    model.save_model(save_to, format="coreml",
                     export_parameters={'prediction_type': 'probability'})

    return cv_val


def train_stage(df_path, lgb_path, xgb_path, cb_path):
    print('>> Loading training data...')
    df = pd.read_csv(df_path)
    print('>> Shape of train data:', df.shape)

    y_df = np.array(df['target'])
    df_ids = np.array(df.index)
    df.drop(['ID_code', 'target'], axis=1, inplace=True)

    lgb_cv_result = np.zeros(df.shape[0])
    xgb_cv_result = np.zeros(df.shape[0])
    cb_cv_result = np.zeros(df.shape[0])

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    print(">> Stratified K Fold with", skf.get_n_splits(df_ids, y_df), "splits")

    print('>> Model Fitting...')
    for counter, ids in enumerate(skf.split(df_ids, y_df)):
        print('>> Fold', counter + 1)
        X_fit, y_fit = df.values[ids[0]], y_df[ids[0]]
        X_val, y_val = df.values[ids[1]], y_df[ids[1]]

        if TRAIN_LGB:
            print('>>> LigthGBM...')
            lgb_cv_result[ids[1]] += fit_lgb(X_fit, y_fit, X_val, y_val, counter, lgb_path, name='lgb')

        if TRAIN_XGB:
            print('>>> XGBoost...')
            xgb_cv_result[ids[1]] += fit_xgb(X_fit, y_fit, X_val, y_val, counter, xgb_path, name='xgb')

        if TRAIN_CB:
            print('>>> CatBoost...')
            cb_cv_result[ids[1]] += fit_cb(X_fit, y_fit, X_val, y_val, counter, cb_path, name='cb')

        del X_fit, X_val, y_fit, y_val
        gc.collect()

    if TRAIN_LGB:
        auc_lgb = round(roc_auc_score(y_df, lgb_cv_result), 4)
        print('>> LightGBM VAL AUC:', auc_lgb)

    if TRAIN_XGB:
        auc_xgb = round(roc_auc_score(y_df, xgb_cv_result), 4)
        print('>> XGBoost  VAL AUC:', auc_xgb)

    if TRAIN_CB:
        auc_cb = round(roc_auc_score(y_df, cb_cv_result), 4)
        print('>> Catboost VAL AUC:', auc_cb)

    if TRAIN_LGB and TRAIN_XGB and TRAIN_CB:
        auc_mean = round(roc_auc_score(y_df, (lgb_cv_result + xgb_cv_result + cb_cv_result) / 3), 4)
        print('>> Mean XGBoost+Catboost+LightGBM, VAL AUC:', auc_mean)

    if TRAIN_LGB and TRAIN_CB:
        auc_mean_lgb_cb = round(roc_auc_score(y_df, (lgb_cv_result + cb_cv_result) / 2), 4)
        print('>> Mean Catboost+LightGBM VAL AUC:', auc_mean_lgb_cb)

    return 0


def prediction_stage(df_path, lgb_path, xgb_path, cb_path):
    print('>> Loading test data...')
    df = pd.read_csv(df_path)
    print('>> Shape of test data:', df.shape)

    df.drop(['ID_code'], axis=1, inplace=True)

    lgb_models = sorted(os.listdir(lgb_path))
    xgb_models = sorted(os.listdir(xgb_path))
    cb_models = sorted(os.listdir(cb_path))

    lgb_result = np.zeros(df.shape[0])
    xgb_result = np.zeros(df.shape[0])
    cb_result = np.zeros(df.shape[0])

    submission = pd.read_csv(SUBMISSIONS_PATH + 'sample_submission.csv')

    print('>> Making predictions...')

    if TRAIN_LGB:
        print('>>> With LightGBM...')
        for m_name in lgb_models:
            # Load LightGBM Model
            model = lgb.Booster(model_file='{}{}'.format(lgb_path, m_name))
            lgb_result += model.predict(df.values)
        lgb_result /= len(lgb_models)
        submission['target'] = lgb_result
        submission.to_csv(SUBMISSIONS_PATH + 'lgb_submission.csv', index=False)

    if TRAIN_XGB:
        print('>>> With XGBoost...')
        for m_name in xgb_models:
            # Load Catboost Model
            model = pickle.load(open('{}{}'.format(xgb_path, m_name), "rb"))
            xgb_result += model.predict_proba(df.values)[:, 1]
        xgb_result /= len(xgb_models)
        submission['target'] = xgb_result
        submission.to_csv(SUBMISSIONS_PATH + 'xgb_submission.csv', index=False)

    if TRAIN_CB:
        print('>>> With CatBoost...')
        for m_name in cb_models:
            # Load Catboost Model
            model = cb.CatBoostClassifier()
            model = model.load_model('{}{}'.format(cb_path, m_name), format='coreml')
            cb_result += model.predict(df.values, prediction_type='Probability')[:, 1]
        cb_result /= len(cb_models)
        submission['target'] = cb_result
        submission.to_csv(SUBMISSIONS_PATH + 'cb_submission.csv', index=False)

    if TRAIN_LGB and TRAIN_XGB and TRAIN_CB:
        submission['target'] = (lgb_result + xgb_result + cb_result) / 3
        submission.to_csv(SUBMISSIONS_PATH + 'xgb_lgb_cb_submission.csv', index=False)

    if TRAIN_CB and TRAIN_LGB:
        submission['target'] = (lgb_result + cb_result) / 2
        submission.to_csv(SUBMISSIONS_PATH + 'lgb_cb_submission.csv', index=False)

    return 0


if __name__ == '__main__':
    start_time = time.time()

    train_path = DATASET_PATH + 'train.csv'
    test_path = DATASET_PATH + 'test.csv'

    lgb_path = MODELS_PATH + 'lgb_models_stack/'
    xgb_path = MODELS_PATH + 'xgb_models_stack/'
    cb_path = MODELS_PATH + 'cb_models_stack/'

    # Create dir for models
    try:
        os.mkdir(lgb_path)
    except: pass
    try:
        os.mkdir(xgb_path)
    except: pass
    try:
        os.mkdir(cb_path)
    except: pass

    print('> Training Stage:')
    train_stage(train_path, lgb_path, xgb_path, cb_path)

    print('> Prediction Stage:')
    prediction_stage(test_path, lgb_path, xgb_path, cb_path)

    end_time = time.time()

    print('> Done.')
    print('> Process took:', end_time - start_time, "seconds")
