import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from imblearn.metrics import geometric_mean_score

# Transform .dat files to .csv files
def to_csv(data, filename, directory):
    output_file = '{}/{}'.format(directory, filename.split('.')[0] + '.csv')
    with open(output_file, 'w') as file:
        for line in data:
            if line.startswith('@input'):
                line = line.split(' ', 1)[1].strip()
                nextline = next(data)
                if nextline.startswith('@output'):
                    nextline = nextline.split(' ', 1)[1]
                    line = line + ', ' + nextline
                    file.write(line.replace(' ', ''))
            elif line.startswith('@'):
                continue
            else:
                file.write(line.replace(' ', ''))

def average(lst):
    return sum(lst)/len(lst)

# Preprocess data
def preprocess(df):
    le = LabelEncoder()
    le.fit(df['Class'])
    df['Class'] = le.transform(df['Class'])

    X = df.drop(['Class'], axis=1).to_numpy()
    y = df['Class'].to_numpy()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

def preprocess_gan(df):
    le = LabelEncoder()
    le.fit(df['Class'])
    df['Class'] = le.transform(df['Class'])
    X = df.drop(['Class'], axis=1).to_numpy()
    y = df['Class'].to_numpy()
    q_trans = QuantileTransformer(n_quantiles=1000, output_distribution='uniform')
    X = q_trans.fit_transform(X)
    scaler = MinMaxScaler(feature_range=(-1, 1), clip=True)
    X = scaler.fit_transform(X)
    
    return X, y

# Perform 5-fold cross validation on a dataset and return metrics
def process(class_alg, balance_alg, df, gan):
    if gan == True:
        X, y = preprocess_gan(df)
    else:
        X, y = preprocess(df)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    folds = kfold.split(X, y)
    f1_scores = []
    gm_scores = []
    auc_scores = []
    for fold in folds:
        X_train = np.take(X, fold[0], axis=0)
        X_test = np.take(X, fold[1], axis=0)
        y_train = np.take(y, fold[0], axis=0)
        y_test = np.take(y, fold[1], axis=0)
        if balance_alg == None:
            X_train_sm, y_train_sm = X_train, y_train
        else :
            X_train_sm, y_train_sm = balance_alg.fit_resample(X_train, y_train)
        class_alg.fit(X_train_sm, y_train_sm)
        y_pred = class_alg.predict(X_test)
        f1_scores.append(f1_score(y_test, y_pred))
        gm_scores.append(geometric_mean_score(y_test, y_pred, average='binary'))
        auc_scores.append(roc_auc_score(y_test, y_pred))

    f1 = np.mean(f1_scores)
    gm = np.mean(gm_scores)
    auc = np.mean(auc_scores)
    return f1, gm, auc

# Process all datasets and calculate average metrics
def process_all(class_alg, balance_alg, directory, gan=False):
    f1_scores = []
    gm_scores = []
    auc_scores = []
    for filename in os.listdir(directory):
        df = pd.read_csv('{}/{}'.format(directory, filename), engine='python')
        f1, gm, auc = process(class_alg, balance_alg, df, gan)
        f1_scores.append(f1)
        gm_scores.append(gm)
        auc_scores.append(auc)
    f1_avg = average(f1_scores)
    gm_avg = average(gm_scores)
    auc_avg = average(auc_scores)

    return f1_avg, gm_avg, auc_avg
