import os
import pickle

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


INPUT_PATH = os.path.dirname(__file__)
train = pd.read_csv(os.path.join(INPUT_PATH, 'data/train.csv'))

print('Train data shape:', train.shape)
print()


X = train.drop(['price_range'], axis=1)
y = train['price_range']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=0
)
print('train, test shape:')
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(
    n_jobs=-1, penalty='none', random_state=0, solver='sag'
)
model.fit(X_train_scaled, y_train)

pipe = Pipeline([('scaler', scaler), ('model', model)])
score = cross_val_score(pipe, X, y, cv=5, n_jobs=-1, scoring='roc_auc_ovr')
print(f'cross validation score(AUC): {score.mean()}')

pickle.dump(model, open(os.path.join(INPUT_PATH, 'model/mobile_price.pkl'), 'wb'))
