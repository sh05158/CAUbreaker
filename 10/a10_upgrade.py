import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker, cm
import random
import os
import sys

print(os.getcwd())
fname_data_train    = './assignment_10_data_train.csv'
fname_data_test     = './assignment_10_data_test.csv'

data_train          = np.genfromtxt(fname_data_train, delimiter=',')
data_test           = np.genfromtxt(fname_data_test, delimiter=',')

number_data_train   = data_train.shape[0]
number_data_test    = data_test.shape[0]

data_train_point    = data_train[:, 0:2]
data_train_point_x  = data_train_point[:, 0]
data_train_point_y  = data_train_point[:, 1]
data_train_label    = data_train[:, 2]

data_test_point     = data_test[:, 0:2]
data_test_point_x   = data_test_point[:, 0]
data_test_point_y   = data_test_point[:, 1]
data_test_label     = data_test[:, 2]

data_train_label_class_0    = (data_train_label == 0)
data_train_label_class_1    = (data_train_label == 1)

data_test_label_class_0     = (data_test_label == 0)
data_test_label_class_1     = (data_test_label == 1)

data_train_point_x_class_0  = data_train_point_x[data_train_label_class_0]
data_train_point_y_class_0  = data_train_point_y[data_train_label_class_0]

data_train_point_x_class_1  = data_train_point_x[data_train_label_class_1]
data_train_point_y_class_1  = data_train_point_y[data_train_label_class_1]

data_test_point_x_class_0   = data_test_point_x[data_test_label_class_0]
data_test_point_y_class_0   = data_test_point_y[data_test_label_class_0]

data_test_point_x_class_1   = data_test_point_x[data_test_label_class_1]
data_test_point_y_class_1   = data_test_point_y[data_test_label_class_1]

print('shape of point in train data = ', data_train_point.shape)
print('shape of point in test data = ', data_train_point.shape)

print('shape of label in train data = ', data_test_label.shape)
print('shape of label in test data = ', data_test_label.shape)

print('data type of point x in train data = ', data_train_point_x.dtype)
print('data type of point y in train data = ', data_train_point_y.dtype)

print('data type of point x in test data = ', data_test_point_x.dtype)
print('data type of point y in test data = ', data_test_point_y.dtype)

try:
    meta = open("./meta2.txt",'a')
except:
    meta = open("./meta2.txt","a")



number_iteration    = 3000 # you can change this value as you want 
learning_rate       = 0.5 # you can change this value as you want 
number_feature      = 6 # you can change this value as you want
alpha               = 0 # you can change this value as you want

def compute_feature(point):

    # ++++++++++++++++++++++++++++++++++++++++++++++++++
    # complete the blanks
    #

    point_x   = point[:, 0]
    point_y   = point[:, 1]
 
    
    feature = np.array([[1, a*x, b*y, c*x**2, d*x**3,e*x**4] for x, y in zip(point_x, point_y)], dtype = float)

    #
    # ++++++++++++++++++++++++++++++++++++++++++++++++++

    return feature

def compute_linear_regression(theta, feature):

    # ++++++++++++++++++++++++++++++++++++++++++++++++++
    # complete the blanks
    #

    value = theta.T@feature.T

    #
    # ++++++++++++++++++++++++++++++++++++++++++++++++++
    
    return value


def sigmoid(z):

    # ++++++++++++++++++++++++++++++++++++++++++++++++++
    # complete the blanks
    #
    
    value = 1/(1+np.exp(-z))

    #
    # ++++++++++++++++++++++++++++++++++++++++++++++++++

    return value 

def compute_logistic_regression(theta, feature):

    # ++++++++++++++++++++++++++++++++++++++++++++++++++
    # complete the blanks
    #

    value = sigmoid(compute_linear_regression(theta, feature))

    #
    # ++++++++++++++++++++++++++++++++++++++++++++++++++

    return value


def compute_residual(theta, feature, label):

    # ++++++++++++++++++++++++++++++++++++++++++++++++++
    # complete the blanks
    #

    residual = np.zeros(shape=(len(label)))

    regression = compute_logistic_regression(theta,feature)

    residual = -label*np.log1p(regression)-(1-label)*np.log1p(1-regression)

    

    #
    # ++++++++++++++++++++++++++++++++++++++++++++++++++

    return residual


def compute_loss(theta, feature, label, alpha):

    # ++++++++++++++++++++++++++++++++++++++++++++++++++
    # complete the blanks
    #

    # loss = np.sum( compute_residual(theta, feature, label) ) / len(label) + (alpha/2) * np.sum(np.power(theta,2))
    loss = np.sum( compute_residual(theta, feature, label) ) / len(label) + (alpha/2) * theta.T@theta

    #
    # ++++++++++++++++++++++++++++++++++++++++++++++++++

    return loss


def compute_gradient(theta, feature, label, alpha):

    # ++++++++++++++++++++++++++++++++++++++++++++++++++
    # complete the blanks
    #

    logistic = compute_logistic_regression(theta, feature)

    gradient = np.zeros(shape=(len(label)))

    gradient = ((logistic - label).T @ feature) / len(label) + alpha*theta


    #
    # ++++++++++++++++++++++++++++++++++++++++++++++++++

    return gradient

def compute_accuracy(theta, feature, label):

    # ++++++++++++++++++++++++++++++++++++++++++++++++++
    # complete the blanks
    #

    logistic = compute_logistic_regression(theta, feature)
    hit = 0

    for currLabel in label[logistic>0.5]:
        if currLabel==1:
            hit+=1
            
    for currLabel in label[logistic<0.5]:
        if currLabel==0:
            hit+=1

    accuracy = hit/len(label)

    #
    # ++++++++++++++++++++++++++++++++++++++++++++++++++

    return accuracy



# number_iteration    = 100000 # you can change this value as you want 
# learning_rate       = 0.1 # you can change this value as you want 
# number_feature      = 5 # you can change this value as you want
# alpha               = 0 # you can change this value as you want

theta                       = np.zeros(number_feature)
loss_iteration_train        = np.zeros(number_iteration)
loss_iteration_test         = np.zeros(number_iteration)
accuracy_iteration_train    = np.zeros(number_iteration)
accuracy_iteration_test     = np.zeros(number_iteration)

maxSum = 0
max1 = 0
max2 = 0

while True:
    # [1, a*x, b*y, c*x**2, d*x**3,e*x**4]
    a = random.randrange(1,101)/10
    b = random.randrange(1,101)/10
    c = random.randrange(1,101)-50
    d = random.randrange(1,101)/10
    e = random.randrange(1,11)/10

    # a = 4444
    # b = 4444
    # c = 
    # d = 
    # e = 

    theta                       = np.zeros(number_feature)
    loss_iteration_train        = np.zeros(number_iteration)
    loss_iteration_test         = np.zeros(number_iteration)
    accuracy_iteration_train    = np.zeros(number_iteration)
    accuracy_iteration_test     = np.zeros(number_iteration)

    feature_train = compute_feature(data_train_point)
    feature_test = compute_feature(data_test_point)
    
    for i in range(number_iteration):

        # ++++++++++++++++++++++++++++++++++++++++++++++++++
        # complete the blanks
        #
    
        

        theta           =  theta - learning_rate*compute_gradient(theta,feature_train, data_train_label, alpha )

        # accuracy_train  =  compute_accuracy(theta, feature_train, data_train_label )
        # accuracy_test   =  compute_accuracy(theta, feature_test, data_test_label )


        #
        # ++++++++++++++++++++++++++++++++++++++++++++++++++

        # loss_iteration_train[i]     = loss_train
        # loss_iteration_test[i]      = loss_test
        # accuracy_iteration_train[i] = accuracy_train
        # accuracy_iteration_test[i]  = accuracy_test

    theta_optimal = theta

    acc1 = compute_accuracy(theta_optimal, feature_train, data_train_label )
    acc2 = compute_accuracy(theta_optimal, feature_test, data_test_label )

    if acc1+acc2 > maxSum:
        maxSum = acc1+acc2
        max1 = acc1
        max2 = acc2

    mystr = str(a)+", "+str(b)+", "+str(c)+", "+str(d)+", "+str(e)+", "+str(acc1)+", "+str(acc2)+", "+str(acc1+acc2)+", "+str(max1)+", "+str(max2)+", "+str(maxSum)
    print( mystr )
    meta.writelines("\n"+mystr)
    meta.flush()