<html><head><meta content="text/html; charset=UTF-8" http-equiv="content-type"></head><body class="c11 c28"><p class="c10"><span class="c3">Adam Standke, </span></p><p class="c10"><span class="c3">CSC 403</span></p><p class="c6"><span class="c5"></span></p><p class="c9"><span class="c17">Using an Artificial Neural Network to mimic a Robot&rsquo;s Internal Navigational System</span></p><p class="c6"><span class="c17"></span></p><p class="c9"><span class="c19 c15">Introduction</span></p><p class="c6"><span class="c5"></span></p><p class="c10"><span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For this project we chose to use data gathered from an autonomous mobile robot as it navigated around a closed room&mdash;as illustrated by figure 1[1]. The data was collected by a team of research scientists that tried to mimic a robot&rsquo;s internal navigational system[1]. They did this by collecting various sensor readings from the robot as it navigated around the room and then applied various machine learning algorithms to the data. Figure 2 shows the robot used in the experiment</span><span class="c20">.</span><span class="c3">This robot had a total of 24 ultrasound sensors around its &lsquo;waist&rsquo; and they were used to read how far away the robot was from the wall[1]. Furthermore, the robot had a laser sensor, a collision sensor, and a robotic head with an omnidirectional head[1].The robot could only move in one of four ways: forward, slight right-turn, sharp right-turn, and slight left-turn. </span></p><p class="c10 c13"><span class="c3">Using the data collected from these sensors, our group will attempt to create a machine learning program that mimics the robot&rsquo;s internal navigational system best.This example is the perfect example for using machine learning models because after inspecting the data, it was really difficult to find patterns from the sensors to predict what movement the robot would make.</span></p><p class="c7 c23 c29"><span>&nbsp; &nbsp; &nbsp; </span><span class="c15">Figure 1</span><span>:</span><span class="c3">&nbsp;The room in which the robot&rsquo;s navigational algorithm was tested</span><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 572.50px; height: 307.03px;"><img alt="" src="images/image3.png" style="width: 572.50px; height: 307.03px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p><p class="c10"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 318.50px; height: 243.46px;"><img alt="" src="images/image6.png" style="width: 318.50px; height: 243.46px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p><p class="c23 c30"><span class="c15">Figure 2:</span><span>&nbsp;The </span><span>SCITOS G5 mobile robot. The robot is programed with a navigational algorithm that uses a wall following technique to avoid collisions.</span><span class="c20">&nbsp; &nbsp;</span></p><p class="c10"><span class="c3">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span></p><p class="c6 c13"><span class="c12"></span></p><p class="c6"><span class="c12"></span></p><p class="c9"><span class="c15 c19">Neural Network</span></p><p class="c6"><span class="c5"></span></p><p class="c10"><span class="c3">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Two neural networks were constructed for this project. Each neural network had similar parameters and characteristics, except the number of hidden layers. Figure 5 shows a neural network with two hidden layers, while figure 6 shows a neural network with one hidden layer. Since most functions can be represented by one or two hidden layers, no more hidden layers were added to the network. Nevertheless, in both network configurations the following steps were taken to construct them. </span></p><p class="c10"><span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Placeholder nodes titled X and y were initially created to store the training data and class labels. After doing so, the construction of the hidden layer(s) was undertaken. For both neural networks, the node titled dnn contained the various connections between each layer. Using </span><span>Tensorflow&rsquo;s</span><span>&nbsp;dense function, a fully connected network was constructed in the following stepwise fashion: the input layer was connected to hidden-layer one, then hidden-layer one was connected to hidden-layer two, and lastly, hidden-layer two was connected to the output layer. </span><span>Tensorflow&rsquo;s</span><span class="c3">&nbsp;dense function not only created and linked all the layers together, but it also, automatically set the initial weights and bias in the network (ie., the kernel matrix was initialized by Tensorflow). </span></p><p class="c10 c13"><span class="c3">After connecting the neural network, the next step was to decide on the type of cost function to use to actually train the model. Since Softmax Regression would be used to &nbsp;predict the label with the highest estimated probability, cross-entropy was chosen as the cost function. The output from the logits was used to compute the mean cross-entropy. This is shown by the node titled loss in figures 5 and 6. </span></p><p class="c10 c13"><span class="c3">Once the network&rsquo;s layers and cost function were constructed, the next step was creating a node that implemented gradient descent. This is shown by the node titled train in figures 5 and 6. The last important step had to do with model evaluation. Overall accuracy was chosen to properly evaluate a model&rsquo;s performance over the training data. This measure determines if the actual target value corresponds to the highest logit. This is represented in figures 5 and 6 by the node eval. &nbsp;</span></p><p class="c23 c26"><span>&nbsp; </span><span class="c15">Figure 5:</span><span class="c3">&nbsp;Artificial Neural Network with two hidden layers</span><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 588.56px; height: 323.50px;"><img alt="" src="images/image5.png" style="width: 838.50px; height: 323.50px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p><p class="c9"><span class="c15">Figure 6:</span><span>&nbsp;Artificial Neural Network with one hidden layer</span><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 615.50px; height: 315.98px;"><img alt="" src="images/image2.png" style="width: 984.80px; height: 315.98px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p><p class="c6"><span class="c5"></span></p><p class="c10"><span class="c3">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;After splitting the original data using a 65&ndash;35 split ratio, the training set consisted of 3,500 training examples and the test set consisted of 1,956 examples. Mini-batch Gradient descent was used for each training step. Furthermore, the batch size used to train the network on each step was set to one hundred and the total number of batches executed during one training epoch was computed using the following formula:</span></p><p class="c10"><span class="c3">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 402.50px; height: 46.97px;"><img alt="" src="images/image4.png" style="width: 402.50px; height: 46.97px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p><p class="c10 c13"><span class="c3">The table in Appendix A shows the results of tuning the neural network. Initially, a network with two hidden layers was used with 300 neurons in the first layer and 150 neurons in the second layer. Furthermore, this initial model had a starting learning rate of 0.01, an epoch of ninety, and each hidden layer used ReLU activation functions. </span></p><p class="c10 c13"><span>After running tensorflow on this initial model, an accuracy score of ninety-four percent was achieved. This meant that the highest scoring logit corresponded to the actual target label ninety-four percent of the time. Even though, this was a high accuracy score, our group wanted to see if this score could be increased by fine-tuning the initial model. As the table in Appendix A shows, initially the number of neurons in each layer was changed. For all intensive purposes this change did not really improve the overall accuracy score by much. Next, our group decided to use only one hidden-layer instead of two. Unfortunately, this change generally decreased the model&rsquo;s overall accuracy score. It was not until our group looked at the academic paper titled: </span><span class="c22">Short-Term Memory Mechanisms in Neural Network Learning of Robot Navigation Tasks: A Case Study</span><span class="c3">, were we able to significantly improve the model&rsquo;s accuracy score. </span></p><p class="c10 c13"><span>&nbsp; &nbsp;In the paper, the authors used a similar neural network to ours, except their neural network used only one hidden layer and used a sigmoid activation function[1]. More importantly, the paper empirically determined that a starting learning rate of 0.05 and an epoch of 500 produced statistically significant results[1]. Thus, the learning rate was increased to 0.05 and the epoch was increased to 200.</span><sup><a href="#ftnt1" id="ftnt_ref1">[1]</a></sup><span class="c3">&nbsp;After doing so, the best accuracy scores were achieved as illustrated by the highlighted results in Appendix A. Furthermore, figure 7 shows that a learning rate of 0.05 lead to a lower cross-entropy score than a learning rate of 0.01. In the end, the three highest scoring models were used for subsequent testing. &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span></p><p class="c23 c31"><span class="c15">Figure 7:</span><span class="c3">&nbsp;Graph that plots cross-entropy versus the number of gradient descent steps for different hyperparameter values. The steeper slopes seen at the beginning of the graph correspond to models that had a higher learning rate.</span><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 710.50px; height: 287.53px;"><img alt="" src="images/image1.png" style="width: 710.50px; height: 287.53px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p><p class="c6"><span class="c5"></span></p><p class="c10"><span class="c24">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="c3">The three best trained models from Appendix A were used on the test set. To evaluate each model&rsquo;s performance, our group decided to use a micro-averaged F1-score, since the target labels had an uneven distribution. Under micro-averaging, precision and recall scores are initially calculated for each target label. Then, all of the precision and recall scores for each label are averaged together to form two different micro averages. The first micro-average represents precision while the other micro-average represents recall. After doing so, the micro-average F1-score is computed by substituting these two micro-averages in the F1-equation for precision and recall. </span></p><p class="c10"><span class="c3">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Table 1 shows the results of testing the three models. Interestingly, each model outputted a micro-averaged F1-score of 89%. The results show that all three models provided a good balance between precision and recall for classifying robotic movement. Furthermore, the results show that this multi-classification task can be accurately modeled with only one hidden layer. Consequently, a deep learning network is not needed for this classification task. &nbsp;</span></p><p class="c6"><span class="c3"></span></p><p class="c9 c16"><span class="c21 c15"></span></p><a id="t.226c2421edb0c0f054dfdd2c992dd39ae01bbba5"></a><a id="t.0"></a><table class="c18"><tbody><tr class="c8"><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c5"># of neurons in hidden layer one</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c5"># of neurons in hidden layer two</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c5"># of hidden layers</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c5"># of gradient descent steps</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c5">Starting Learning Rate</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c5">Activation function </span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c5">Micro-Averaged F1 score </span></p></td></tr><tr class="c8"><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">200</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">1</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">7000</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.05</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">89%</span></p></td></tr><tr class="c8"><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">200</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">100</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">2</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">7000</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.05</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">89%</span></p></td></tr><tr class="c8"><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">200</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">100</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">2</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">7000</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.05</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">Tanh</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">89%</span></p></td></tr></tbody></table><p class="c9"><span class="c12">Table 1</span></p><p class="c6"><span class="c3"></span></p><p class="c10"><span class="c3">&nbsp; &nbsp; &nbsp; &nbsp; </span></p><p class="c10"><span class="c3">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span></p><p class="c9"><span class="c19 c15">Citations</span></p><p class="c6"><span class="c3"></span></p><p class="c10 c27"><span>[1] </span><span class="c25">Freire, Ananda L., et al. &ldquo;Short-Term Memory Mechanisms in Neural Network Learning of Robot Navigation Tasks: A Case Study.&rdquo; </span><span class="c22 c25">2009 6th Latin American Robotics Symposium (LARS 2009)</span><span class="c25">, 2009, doi:10.1109/lars.2009.5418323.</span></p><p class="c6"><span class="c3"></span></p><p class="c9"><span class="c19 c15">Appendix A</span></p><p class="c6"><span class="c5"></span></p><p class="c9"><span class="c15 c21">Fine-Tuning the ANN Model</span></p><a id="t.3ad3298349bf6eb746c7db606d319ba463a1cdf4"></a><a id="t.1"></a><table class="c18"><tbody><tr class="c8"><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c5"># of neurons in hidden layer one</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c5"># of neurons in hidden layer two</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c5"># of hidden layers</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c5"># of gradient descent steps</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c5">Starting Learning Rate</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c5">Activation function</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c5">Overall Accuracy Score</span></p></td></tr><tr class="c8"><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">300</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">150</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">2</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">3495</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.01</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">94%</span></p></td></tr><tr class="c8"><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">250</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">150</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">2</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">3495</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.01</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">95%</span></p></td></tr><tr class="c8"><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">200</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">150</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">2</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">3495</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.01</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">92%</span></p></td></tr><tr class="c8"><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">150</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">150</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">2</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">3495</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.01</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">93%</span></p></td></tr><tr class="c8"><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">150</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">0</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">1</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">3495</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.01</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">88%</span></p></td></tr><tr class="c8"><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">200</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">0</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">1</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">3495</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.01</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">85%</span></p></td></tr><tr class="c8"><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">250</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">0</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">1</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">3495</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.01</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">90%</span></p></td></tr><tr class="c8"><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">300</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">0</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">1</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">3495</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.01</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">87%</span></p></td></tr><tr class="c8"><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">350</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">0</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">1</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">3495</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.01</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">88%</span></p></td></tr><tr class="c8"><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">400</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">0</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">1</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">3495</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.01</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">88%</span></p></td></tr><tr class="c8"><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">500</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">0</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">1</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">3495</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.01</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">84%</span></p></td></tr><tr class="c8"><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">600</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">0</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">1</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">3495</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.01</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">87%</span></p></td></tr><tr class="c8"><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">350</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">150</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">2</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">3495</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.01</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">93%</span></p></td></tr><tr class="c8"><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">400</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">150</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">2</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">3495</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.01</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">94%</span></p></td></tr><tr class="c8"><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">500</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">150</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">2</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">3495</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.01</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">93%</span></p></td></tr><tr class="c8"><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">600</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">150</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">2</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">3495</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.01</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">94%</span></p></td></tr><tr class="c8"><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">300</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">100</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">2</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">3495</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.01</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">95%</span></p></td></tr><tr class="c8"><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">300</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">50</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">2</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">3495</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.01</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">92%</span></p></td></tr><tr class="c8"><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">300</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">120</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">2</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">3495</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.01</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">93%</span></p></td></tr><tr class="c8"><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">300</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">140</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">2</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">3495</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.01</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">94%</span></p></td></tr><tr class="c8"><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">300</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">300</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">2</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">3495</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.01</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">95%</span></p></td></tr><tr class="c8"><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">400</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">400</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">2</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">3495</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.01</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">93%</span></p></td></tr><tr class="c8"><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">400</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">200</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">2</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">3495</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.01</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">93%</span></p></td></tr><tr class="c8"><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">500</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">200</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">2</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">3495</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.01</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">95%</span></p></td></tr><tr class="c8"><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">500</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">250</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">2</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">3495</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.01</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">94%</span></p></td></tr><tr class="c8"><td class="c0 c14" colspan="1" rowspan="1"><p class="c2"><span class="c1">200</span></p></td><td class="c0 c14" colspan="1" rowspan="1"><p class="c2"><span class="c1">0</span></p></td><td class="c0 c14" colspan="1" rowspan="1"><p class="c2"><span class="c1">1</span></p></td><td class="c0 c14" colspan="1" rowspan="1"><p class="c2"><span class="c1">7000</span></p></td><td class="c4 c14" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.05</span></p></td><td class="c4 c14" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4 c14" colspan="1" rowspan="1"><p class="c7"><span class="c3">98%</span></p></td></tr><tr class="c8"><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">200</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">0</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">1</span></p></td><td class="c0" colspan="1" rowspan="1"><p class="c2"><span class="c1">7000</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.05</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">Sigmoid</span></p></td><td class="c4" colspan="1" rowspan="1"><p class="c7"><span class="c3">75%</span></p></td></tr><tr class="c8"><td class="c0 c11" colspan="1" rowspan="1"><p class="c2"><span class="c1">200</span></p></td><td class="c0 c11" colspan="1" rowspan="1"><p class="c2"><span class="c1">0</span></p></td><td class="c0 c11" colspan="1" rowspan="1"><p class="c2"><span class="c1">1</span></p></td><td class="c0 c11" colspan="1" rowspan="1"><p class="c2"><span class="c1">7000</span></p></td><td class="c4 c11" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.05</span></p></td><td class="c4 c11" colspan="1" rowspan="1"><p class="c7"><span class="c3">Tanh</span></p></td><td class="c4 c11" colspan="1" rowspan="1"><p class="c7"><span class="c3">96%</span></p></td></tr><tr class="c8"><td class="c0 c14" colspan="1" rowspan="1"><p class="c2"><span class="c1">200</span></p></td><td class="c0 c14" colspan="1" rowspan="1"><p class="c2"><span class="c1">100</span></p></td><td class="c0 c14" colspan="1" rowspan="1"><p class="c2"><span class="c1">2</span></p></td><td class="c0 c14" colspan="1" rowspan="1"><p class="c2"><span class="c1">7000</span></p></td><td class="c4 c14" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.05</span></p></td><td class="c4 c14" colspan="1" rowspan="1"><p class="c7"><span class="c3">ReLu</span></p></td><td class="c4 c14" colspan="1" rowspan="1"><p class="c7"><span class="c3">100%</span></p></td></tr><tr class="c8"><td class="c0 c14" colspan="1" rowspan="1"><p class="c2"><span class="c1">200</span></p></td><td class="c0 c14" colspan="1" rowspan="1"><p class="c2"><span class="c1">100</span></p></td><td class="c0 c14" colspan="1" rowspan="1"><p class="c2"><span class="c1">2</span></p></td><td class="c0 c14" colspan="1" rowspan="1"><p class="c2"><span class="c1">7000</span></p></td><td class="c4 c14" colspan="1" rowspan="1"><p class="c7"><span class="c3">0.05</span></p></td><td class="c4 c14" colspan="1" rowspan="1"><p class="c7"><span class="c3">Tanh</span></p></td><td class="c4 c14" colspan="1" rowspan="1"><p class="c7"><span class="c3">100%</span></p></td></tr></tbody></table><p class="c9 c16"><span class="c12"></span></p><p class="c9 c16"><span class="c12"></span></p><p class="c9 c16"><span class="c12"></span></p><p class="c6"><span class="c3"></span></p><div><p class="c6"><span class="c3"></span></p><p class="c6"><span class="c3"></span></p></div><hr class="c32"><div><p class="c7 c23"><a href="#ftnt_ref1" id="ftnt1">[1]</a><span class="c20">&nbsp;As </span><span>a side note, the reason an epoch of 200 was used rather than an epoch of 500, was due to that fact that it took less than 500 epochs for the model to converge to a global minimum when activation functions such as the hyperbolic tangent function (ie., Tanh) or the ReLU function were used</span></p></div></body></html>
