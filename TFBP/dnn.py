import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


BASE = ['A', 'C', 'G', 'T']
le = preprocessing.LabelEncoder()
le.fit(BASE)

data_df = pd.read_csv('train.csv', index_col='id')

X, Y = [], []
for row in data_df.iterrows():
    data = row[1]
    X.append(le.transform(list(data['sequence'])))
    Y.append(data['label'])
X_encoded = np.array(X)
Y_encoded = np.array(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y_encoded, test_size=0.1, random_state=42)
X_train_oh = tf.layers.flatten(tf.one_hot(X_train, depth=4))
X_test_oh = tf.layers.flatten(tf.one_hot(X_test, depth=4))

submit_df = pd.read_csv('test.csv', index_col='id')
X_kaggle = []
for row in submit_df.iterrows():
    data = row[1]
    X_kaggle.append(le.transform(list(data['sequence'])))
X_kaggle_enc = np.array(X_kaggle)

X_kaggle_oh = tf.layers.flatten(tf.one_hot(X_kaggle_enc, depth=4))

n_inputs = len(data_df['sequence'][0]) * len(BASE)
n_hidden1 = 30
n_hidden2 = 10
n_outputs = 2

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=None, name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.contrib.layers.fully_connected(X, n_hidden1, scope="hidden1")
    hidden2 = tf.contrib.layers.fully_connected(hidden1, n_hidden2, scope="hidden2")
    logits = tf.contrib.layers.fully_connected(hidden2, n_outputs, scope="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver

n_epochs = 5000

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        sess.run(
            training_op,
            feed_dict={
                X: sess.run(X_train_oh),
                y: Y_train
            })
        acc_train = accuracy.eval(feed_dict={
            X: sess.run(X_train_oh),
            y: Y_train
        })
        acc_test = accuracy.eval(feed_dict={
            X: sess.run(X_test_oh),
            y: Y_test
        })
        if epoch % 500 == 0:
            print(epoch, "train acc:", acc_train, "test acc:", acc_test)
    Z = logits.eval(feed_dict={X: sess.run(X_kaggle_oh)})
    y_pred = np.argmax(Z, axis=1)

print("Saving prediction into submit.csv...")
pd.DataFrame({'prediction': y_pred}).to_csv("submit.csv", index_label="id")
