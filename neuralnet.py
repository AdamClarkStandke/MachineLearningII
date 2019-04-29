#neural network Softmax regression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from datetime import datetime

#creates log directory for tensorboard
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "nn_logs"
logdir = "{}/run--{}".format(root_logdir,now)

#loads the robot data
def load_robot_data():
     ifile = open("phpVeNa5j.csv", "rb")
     return pd.read_csv(ifile)

#creates a mini-batch for gradient descent
def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)  
    indices = np.random.randint(m, size=batch_size)
    X_batch = X_train[indices] 
    y_batch = y_train.reshape(-1)[indices]
    return X_batch, y_batch


#getting data and spliting it(have to work on this )
robot = load_robot_data()
y = robot["Class"]
X = robot[["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24" ]]
y= y.to_numpy()
X= X.to_numpy()

#scaling the data before feeding it to neural net
scaler = StandardScaler()
X = scaler.fit_transform(X)
#Splitting data into train and test (probably need to change due to it not being alot of data)
X_train, X_test, y_train, y_test = X[:2000], X[2000:], y[:2000], y[2000:]
#gets the shape of the training set
m, n = X_train.shape


#Construction of neural net

#number of features
n_inputs = 24
#number of hidden neurons in layer 1 (can tweak for extra performacne)
n_hidden1= 300
#number of hidden neurons in layer 2 (can tweak for extra performance)
n_hidden2=100
#number of outputs 
n_outputs=5
#how long to train for
n_epochs = 100
#learning rate (can tweak for extra performance)
learning_rate = 0.01
#how many instaces to feed the network on each training step (can tweak for performance)
batch_size = 100
n_batches = int(np.ceil(m/batch_size))



X=tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y=tf.placeholder(tf.int64, shape=(None), name="y")
with tf.name_scope("dnn"): #links the layers of nerual net
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
with tf.name_scope("loss"): #uses cross entropy for loss function
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
with tf.name_scope("train"): #trains the neural net
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
with tf.name_scope("eval"): #evaulates the output before entering softmax regression
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()
# saver = tf.train.Saver() #saves the model 
loss_summary = tf.summary.scalar("loss", loss) #creates node for evaluation of loss func on tenserboard
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())


#execution of neural net
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs): #how long to train for
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size) #fetches each mini-batch
            if batch_index % 10 == 0: #prints out step statistics to tenserboard
                summary_step = loss_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_step, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        print(epoch, "Train accuracy:", acc_train) # prints out accuracy of output nodes of neural net(ie how good the output nodes performing)



