import numpy as np
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split

data = pd.read_csv('train.csv', index_col='id')
train, test = train_test_split(data, test_size=0.99)
le = preprocessing.LabelEncoder()
le.fit(['A', 'G', 'C', 'T'])

X_train, Y_train = [], []
X_test, Y_test = [], []

for row in train.iterrows():
    data = row[1]
    X_train.append(le.transform(list(data['sequence'])))
    Y_train.append(data['label'])
X_train_array = np.array(X_train)
Y_train_array = np.array(Y_train)

for row in test.iterrows():
    data = row[1]
    X_test.append(le.transform(list(data['sequence'])))
    Y_test.append(data['label'])
X_test_array = np.array(X_test)
Y_test_array = np.array(Y_test)

svc = svm.SVC(kernel='rbf')
svc.fit(X_train_array, Y_train_array)

kaggle_data = pd.read_csv('test.csv', index_col='id')
X_kaggle = []

for row in kaggle_data.iterrows():
    data = row[1]
    X_kaggle.append(le.transform(list(data['sequence'])))
X_kaggle_array = np.array(X_kaggle)

Y_kaggle = svc.predict(X_kaggle_array)

pd.DataFrame({
    'prediction': Y_kaggle
}).to_csv("submit.csv", index_label="id")
