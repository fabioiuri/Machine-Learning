"""
Tree Boosting Regressor using multiple boosting techniques and stacking their results by taking
a weighted average of each models' predictions.

Used to solve HackerEarth Lunar Eclipse Duration problem.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import pickle
import os
import gc
import time
import joblib
from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

gc.enable()


''' CONFIGURABLE PARAMETERS '''
NUM_FOLDS = 5

TRAIN_LGB = True
TRAIN_XGB = False
TRAIN_CB = False

LGB_USE_GPU = False
XGB_USE_GPU = False
CB_USE_GPU = False

MODELS_PATH = './models/'
SUBMISSIONS_PATH = './submissions/'
DATASET_PATH = './dataset/'
''' CONFIGURABLE PARAMETERS '''


def preprocess_data(df, predict=False):
    X = df.values

    eclipse_type_arrays = {
        'A': [], 'H': [], 'N': [], 'P': [], 'T': []
    }

    # Feature extraction
    for x in X:
        # date and time
        year, month = [int(s) for s in x[0].split("-")]
        day = x[1]
        hour, minute, second = [int(s) for s in x[2].split(":")]
        try:
            x[0] = datetime(year, month, day, hour, minute, second).timestamp()
            x[1] = datetime(year, month, day, hour, minute, second).weekday()
        except:
            x[0] = datetime(year, month, day-1, hour, minute, second).timestamp()
            x[1] = datetime(year, month, day-1, hour, minute, second).weekday()
        # latitude
        if 'S' in x[8]:
            x[8] = -float(x[8].replace('S', ''))
        else:
            x[8] = +float(x[8].replace('N', ''))
        # longitude
        if 'W' in x[9]:
            x[9] = -float(x[9].replace('W', ''))
        else:
            x[9] = +float(x[9].replace('E', ''))
        # eclipse type
        for key in eclipse_type_arrays:
            eclipse_type_arrays[key].append(1 if key in x[5] else 0)

    X = np.delete(X, np.s_[1:3], axis=1)  # delete day and time columns (results are stored on the date column)
    X = np.delete(X, np.s_[3:4], axis=1)  # delete the categorical column to insert it later one-hot encoded

    # Feature scaling
    if predict:
        scaler_x = joblib.load('./scalers/scaler_x.pkl')
        X = scaler_x.transform(X)
    else:
        scaler_x = (MinMaxScaler()).fit(X)
        X = scaler_x.transform(X)
        joblib.dump(scaler_x, './scalers/scaler_x.pkl')

    # insert one hot encoded eclipse types
    for key in eclipse_type_arrays:
        X = np.column_stack((X, eclipse_type_arrays[key]))

    return X


def fit_rf(X_fit, y_fit, X_val, y_val, counter, rf_path, name):
    model = lgb.LGBMRegressor(max_depth=12,
                               n_estimators=999999,
                               n_jobs=-1,
                               boosting_type='rf',
                               bagging_freq=2,
                               bagging_fraction=0.9,
                               num_leaves=50,
                               device=("gpu" if LGB_USE_GPU else "cpu"))

    model.fit(X_fit, y_fit,
              eval_set=[(X_val, y_val)],
              verbose=0,
              early_stopping_rounds=1000)

    cv_val = model.predict(X_val)

    # Save LightGBM Model
    save_to = '{}{}_fold{}.txt'.format(rf_path, name, counter + 1)
    model.booster_.save_model(save_to)

    return cv_val


def fit_lgb(X_fit, y_fit, X_val, y_val, counter, lgb_path, name):
    model = lgb.LGBMRegressor(max_depth=3,
                              n_estimators=999999,
                               n_jobs=-1,
                               device=("gpu" if LGB_USE_GPU else "cpu"))

    model.fit(X_fit, y_fit,
              eval_set=[(X_val, y_val)],
              verbose=0,
              early_stopping_rounds=1000)

    cv_val = model.predict(X_val)

    # Save LightGBM Model
    save_to = '{}{}_fold{}.txt'.format(lgb_path, name, counter + 1)
    model.booster_.save_model(save_to)

    return cv_val


def fit_xgb(X_fit, y_fit, X_val, y_val, counter, xgb_path, name):
    model = xgb.XGBRegressor(max_depth=3,
                                n_estimators=999999,
                                n_jobs=-1,
                                objective= "reg:squarederror",
                                gpu_hist=XGB_USE_GPU)

    model.fit(X_fit, y_fit,
              eval_set=[(X_val, y_val)],
              verbose=False,
              early_stopping_rounds=1000)

    # plot_importance(model)
    # plt.show()

    cv_val = model.predict(X_val)

    # Save XGBoost Model
    save_to = '{}{}_fold{}.dat'.format(xgb_path, name, counter + 1)
    pickle.dump(model, open(save_to, "wb"))

    return cv_val


def fit_cb(X_fit, y_fit, X_val, y_val, counter, cb_path, name):
    model = cb.CatBoostRegressor(iterations=999999,
                                  max_depth=3,
                                  thread_count=-1,
                                  task_type=("GPU" if CB_USE_GPU else "CPU"))

    model.fit(X_fit, y_fit,
              eval_set=[(X_val, y_val)],
              verbose=0, early_stopping_rounds=1000)

    cv_val = model.predict(X_val)

    # Save Catboost Model
    save_to = "{}{}_fold{}.mlmodel".format(cb_path, name, counter + 1)
    model.save_model(save_to, format="coreml")

    return cv_val


def train_stage(df_path, lgb_path, xgb_path, cb_path):
    print('>> Loading training data...')
    df = pd.read_csv(df_path)

    y_df = np.array(df['Eclipse Duration (m)'])
    df_ids = np.array(df.index)
    df.drop(['Eclipse Duration (m)', 'Lunation Number'], axis=1, inplace=True)

    X = preprocess_data(df)
    print('>> Shape of train data:', X.shape)

    lgb_cv_result = np.zeros(df.shape[0])
    xgb_cv_result = np.zeros(df.shape[0])
    cb_cv_result = np.zeros(df.shape[0])

    skf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    print(">> K Fold with", skf.get_n_splits(df_ids, y_df), "splits")

    print('>> Model Fitting...')
    for counter, ids in enumerate(skf.split(df_ids, y_df)):
        print('>> Fold', counter + 1)
        X_fit, y_fit = X[ids[0]], y_df[ids[0]]
        X_val, y_val = X[ids[1]], y_df[ids[1]]

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
        rmse_lgb = round(sqrt(mean_squared_error(y_df, lgb_cv_result)), 4)
        print('>> LightGBM VAL RMSE:', rmse_lgb)

    if TRAIN_XGB:
        rmse_xgb = round(sqrt(mean_squared_error(y_df, xgb_cv_result)), 4)
        print('>> XGBoost  VAL RMSE:', rmse_xgb)

    if TRAIN_CB:
        rmse_cb = round(sqrt(mean_squared_error(y_df, cb_cv_result)), 4)
        print('>> Catboost VAL RMSE:', rmse_cb)

    if TRAIN_LGB and TRAIN_XGB and TRAIN_CB:
        rmse_mean = round(sqrt(mean_squared_error(y_df, (lgb_cv_result + xgb_cv_result + cb_cv_result) / 3)), 4)
        print('>> Mean XGBoost+Catboost+LightGBM, VAL RMSE:', rmse_mean)

    if TRAIN_LGB and TRAIN_CB:
        rmse_mean_lgb_cb = round(sqrt(mean_squared_error(y_df, (lgb_cv_result + cb_cv_result) / 2)), 4)
        print('>> Mean Catboost+LightGBM VAL RMSE:', rmse_mean_lgb_cb)

    if TRAIN_LGB and TRAIN_XGB:
        rmse_mean_lgb_xgb = round(sqrt(mean_squared_error(y_df, (lgb_cv_result + xgb_cv_result) / 2)), 4)
        print('>> Mean XGBoost+LightGBM VAL RMSE:', rmse_mean_lgb_xgb)

    return 0


def prediction_stage(df_path, lgb_path, xgb_path, cb_path):
    print('>> Loading test data...')
    df = pd.read_csv(df_path)

    df.drop(['ID', 'Lunation Number'], axis=1, inplace=True)

    lgb_models = sorted(os.listdir(lgb_path))
    xgb_models = sorted(os.listdir(xgb_path))
    cb_models = sorted(os.listdir(cb_path))

    lgb_result = np.zeros(df.shape[0])
    xgb_result = np.zeros(df.shape[0])
    cb_result = np.zeros(df.shape[0])

    submission = pd.read_csv(SUBMISSIONS_PATH + 'sample_submission.csv')

    X = preprocess_data(df, predict=True)
    print('>> Shape of test data:', X.shape)

    print('>> Making predictions...')

    if TRAIN_LGB:
        print('>>> With LightGBM...')
        for m_name in lgb_models:
            # Load LightGBM Model
            model = lgb.Booster(model_file='{}{}'.format(lgb_path, m_name))
            lgb_result += model.predict(X)
        lgb_result /= len(lgb_models)
        submission['Eclipse Duration (m)'] = lgb_result
        submission.to_csv(SUBMISSIONS_PATH + 'lgb_submission.csv', index=False)

    if TRAIN_XGB:
        print('>>> With XGBoost...')
        for m_name in xgb_models:
            # Load Catboost Model
            model = pickle.load(open('{}{}'.format(xgb_path, m_name), "rb"))
            xgb_result += model.predict(X)
        xgb_result /= len(xgb_models)
        submission['Eclipse Duration (m)'] = xgb_result
        submission.to_csv(SUBMISSIONS_PATH + 'xgb_submission.csv', index=False)

    if TRAIN_CB:
        print('>>> With CatBoost...')
        for m_name in cb_models:
            # Load Catboost Model
            model = cb.CatBoostRegressor()
            model = model.load_model('{}{}'.format(cb_path, m_name), format='coreml')
            cb_result += model.predict(X)
        cb_result /= len(cb_models)
        submission['Eclipse Duration (m)'] = cb_result
        submission.to_csv(SUBMISSIONS_PATH + 'cb_submission.csv', index=False)

    if TRAIN_LGB and TRAIN_XGB and TRAIN_CB:
        submission['Eclipse Duration (m)'] = (lgb_result + xgb_result + cb_result) / 3
        submission.to_csv(SUBMISSIONS_PATH + 'xgb_lgb_cb_submission.csv', index=False)

    if TRAIN_CB and TRAIN_LGB:
        submission['Eclipse Duration (m)'] = (lgb_result + cb_result) / 2
        submission.to_csv(SUBMISSIONS_PATH + 'lgb_cb_submission.csv', index=False)

    if TRAIN_XGB and TRAIN_LGB:
        submission['Eclipse Duration (m)'] = (lgb_result + xgb_result) / 2
        submission.to_csv(SUBMISSIONS_PATH + 'xgb_lgb_submission.csv', index=False)

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
    try:
        os.mkdir(rf_path)
    except: pass

    print('> Training Stage:')
    train_stage(train_path, lgb_path, xgb_path, cb_path)

    print('> Prediction Stage:')
    prediction_stage(test_path, lgb_path, xgb_path, cb_path)

    end_time = time.time()

    print('> Done.')
    print('> Process took:', end_time - start_time, "seconds")
