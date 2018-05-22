#The future statements are present for Python files to be compatible
#with both Python 2 and Python 3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


#Imports that are required
#numpy is used for the mathematical calculations
import numpy as np
import tensorflow as tf
import time
import data_helpers


# A timer is started that will record the runtime
beginTime=time.time()

# Parameters for our dataset is defined here
batch_size=100
learning_rate=0.005
max_steps=1000

# Data is loaded into the script
# load_data() is a function in the data_helpers script that
# splits our dataset into two, the training set and the test set
# The training set consists of 50,000 images while the test set consists 10,000 images
data_sets=data_helpers.load_data()
# load_data returns images_train and labels train which are the 50000 images and their labels for training
# also returns the images_test and labels_test which are the 10000 images and their labels for testing
# classes are returned which have 10 text labels that translate numerical value to word(0 for plane, 1 for car)


# Defining the placeholders
# Tensorflow uses a C++ backend to do the numerical computations. We first define all the calculations
# by building a Tensorflow graph. No actual calculation happens now but the formulas are set
# image_placeholders are where our input data will go. Currently it doesn't have any data but the shape of the data that will come is defined
# The shape argument defines the inputs dimensions. As we are going to provide the model with a lot of data therefore
# None defines that the shape can be of any length while each image will be of 3072 values
# The labels placeholders contains integer values from 0 to 1 representing each class.
# Similarly as we don't want to specify how many images we will give therefore the length is None
images_placeholder=tf.placeholder(tf.float32,shape=[None, 3072])
labels_placeholder=tf.placeholder(tf.int64,shape=[None])

# Defining the variables that we will optimize
# We look at each pixel individually and evaluate the chance of it belonging to a certain class.
# If a pixel is red then the weight for it to be of the car class will be positive
# and will multiply with the pixel value to increase the score
# At the end we look at the highest score and determine the label


# The line below says that there are 3072 x 10 values as weights which are 0 in the beginning
# The bias is a 10 dimensional vector which allows us to start with non-zero classes
weights=tf.Variable(tf.zeros([3072,10]))
biases=tf.Variable(tf.zeros([10]))

# Our image is a 3072-dimensional vector so if
# we multiply the vector with a 3072x10 matrix of weights, the result will be a 10 dimensional vector

# Defining the classifiers result
# Multiple images are evaluated in a single step
# The prediction takes place in this statement
logits=tf.matmul(images_placeholder,weights)+biases

# Loss Function
# We first input the training data and let the model make predictions, the predictions are then compared with the correct class labels
# The numerical result of the comparison is the loss
# The smaller the loss the more accurate our results

# logits contains arbitary real numbers.
# At first we will convert the numbers into probabilites (0 to 1)
# This is done using the softmax function.
# We then compare the probability distribution to the true probability
# distribution where the correct class has a probability of 1 and all others are 0
# Cross entropy is used to compare the two distributions
loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels_placeholder))

# We then calculate the average loss value
# Tensorflow changes our parameters using auto differentiation where it calculates the gradient of the loss
# It then evaluates the parameter's influence on the overall loss and adjusts all parameter values accordingly
# Tensorflow uses gradient descent which looks at the model's current state before updating the parameters.
# It doesn't look at the past records. Gradient descent requires the learning rate which defines how much the parameters will change
# If the learning rate is too big, then the parameters might overshoot
train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


# Calculating model's accuracy
# Compare prediction with true label
# tf.equal returns a vector of boolean values
correct_predicton=tf.equal(tf.argmax(logits,1), labels_placeholder)
# Calculate accuracy of the prediction
# The boolean values are then cast to float values either 0 or 1
accuracy=tf.reduce_mean(tf.cast(correct_predicton,tf.float32))

#=============================================
#Running our Tensorflow graph
#=============================================

with tf.Session() as sess:
    #Initialize variables
    sess.run(tf.global_variables_initializer())
    #Repeat max_steps times
    for i in range(max_steps):
        # The following lines randomly pick a certain number of images during each iteration
        # The chosen images and labels are known as batches
        indices=np.random.choice(data_sets['images_train'].shape[0], batch_size)
        images_batch=data_sets['images_train'][indices]
        labels_batch=data_sets['labels_train'][indices]
        if i%100==0:
            # After every 100 iterations we check the models accuracy. We will call the accuracy formula defined above
            # We also assign the images_batch to our images_placeholder variable
            train_accuracy=sess.run(accuracy, feed_dict={images_placeholder:images_batch,labels_placeholder:labels_batch})
            print('Step {:5d}: training accuracy {:g}'.format(i, train_accuracy))
            #We now ask the model to perform the training step
            sess.run(train_step, feed_dict={images_placeholder:images_batch, labels_placeholder:labels_batch})
    # After training completes on the batch we check the accuracy by calling the test_accuracy formula
    # Here the images from the test part are passed onto the place holders and the test results come. The images are first seen here by the model
    test_accuracy=sess.run(accuracy, feed_dict={images_placeholder:data_sets['images_test'], labels_placeholder:data_sets['labels_test']})
    print('Test accuracy {:g}'.format(test_accuracy))

endTime=time.time()
print('Total time {:5.2f}s'.format(endTime-beginTime))
