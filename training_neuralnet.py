#created by Adam Standke
#neural network using Softmax regression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.metrics import f1_score

#creates a log directory for tensorboard based on the current Y/m/d/time
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "nn_logs"
logdir = "{}/run--{}".format(root_logdir,"me5three")

#loads the robot data into a pd dataframe
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


#gets data and splits it based on taget and attributes
robot = load_robot_data()
y = robot["Class"]
X = robot[["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24" ]]

#converts dataframe to numpy array
y= y.to_numpy()
X= X.to_numpy()

#scaling the data before inputting it into neural net with zero mean and unit varience
scaler = StandardScaler()
X = scaler.fit_transform(X)
#Splits data into train and test
X_train, X_test, y_train, y_test = X[:3500], X[3500:], y[:3500], y[3500:]
#gets the shape of the training set
m, n = X_train.shape


#Actual Construction of neural net

#number of features
n_inputs = 24
#number of hidden neurons in layer 1 
n_hidden1= 200
#number of hidden neurons in layer 2 
n_hidden2=100
#number of outputs 
n_outputs=5
#how long to train for
n_epochs = 200 #200, 90
#starting learning rate 
learning_rate = 0.05 #.01 and .05
#how many instaces to feed into the network for each training step 
batch_size = 100
n_batches = int(np.ceil(m/batch_size))


#creates two placeholder nodes that will input the minibatches into the network
X=tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y=tf.placeholder(tf.int64, shape=(None), name="y")
with tf.name_scope("dnn"): #links the layers of nerual net (neural net has two hidden latyers)
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
with tf.name_scope("loss"): #uses cross entropy for the loss function
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
with tf.name_scope("train"): #unit that trains the neural net
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
with tf.name_scope("eval"): #evaulates the output before entering softmax regression portion of neural net
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver() #creates a saver unit in graph 
loss_summary = tf.summary.scalar("loss", loss) #creates unit for evaluation of loss function on tenserboard
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph()) #writes information of graph into the log directory


#execution of neural net
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs): #portion of code that trains the model
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size) #fetches each mini-batch
            if batch_index % 10 == 0: #prints out step statistics to tenserboard
                summary_step = loss_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_step, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch}) #trains the neural network 
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch}) #evaluates the accuracy of the nerual net 
    print(epoch, "Train accuracy:", acc_train) # prints out the overall accuracy model
    save_path = saver.save(sess, "/Users/adam/Desktop/Project_ML/neuralNet_savemodel.ckpt") #saves the trained graph for later testing
    



