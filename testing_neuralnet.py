#created by Adam Standke
#code that test's the neural network
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.metrics import f1_score

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


#getting data and spliting it
robot = load_robot_data()
y = robot["Class"]
X = robot[["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24" ]]
y= y.to_numpy()
X= X.to_numpy()

#scaling the data before feeding it to neural net
scaler = StandardScaler()
X = scaler.fit_transform(X)
#Splitting data into train and test 
X_train, X_test, y_train, y_test = X[:3500], X[3500:], y[:3500], y[3500:]
#gets the shape of the training set
m, n = X_train.shape


#Construction of neural net

#number of features
n_inputs = 24
#number of hidden neurons in layer 1 
n_hidden1= 200
#number of hidden neurons in layer 2 
n_hidden2=100
#number of outputs 
n_outputs=5
#how long to train for
n_epochs = 90
#learning rate (can tweak for extra performance)
learning_rate = 0.05
#how many instaces to feed the network on each training step (can tweak for performance)
batch_size = 100
n_batches = int(np.ceil(m/batch_size))



X=tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y=tf.placeholder(tf.int64, shape=(None), name="y")
with tf.name_scope("dnn"): #links the layers of nerual net
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
with tf.name_scope("loss"): #uses cross entropy for the loss function
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
with tf.name_scope("train"): #trains the neural net
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
with tf.name_scope("eval"): #evaulates the output before entering softmax regression
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver() #creates a save unit in graph 
loss_summary = tf.summary.scalar("loss", loss) #creates unit for evaluation of loss function on tenserboard
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())#writes information of graph into the log directory


#tests the saved neural network by restoring the trained graph
with tf.Session() as sess:
    saver.restore(sess, "/Users/adam/Desktop/Project_ML/neuralNet_savemodel.ckpt")
    Z=logits.eval(feed_dict={X: X_test}) #input the the test set into the neural network 
    y_pred = np.argmax(Z, axis=1) #contains the predicted values from the neural network
    print(f1_score(y_test, y_pred, average='weighted')) #compare predicted values to acutal values and output f1 score