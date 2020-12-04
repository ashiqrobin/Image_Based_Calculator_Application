import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import numpy as np
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

x_test = []
y_test = []
count = 1

mnist_model = load_model("/Users/mertaydayanc/Desktop/imageCalculatorVenv/results/new_model.h5")

with open('mnist_test.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row  in csv_reader:
      temp=row[1:]
      #temp=np.reshape(temp,(784,1))
      x_test.append(temp)
      if(row[0] == '+'):
        y_test.append('10')
      elif(row[0] == '-'):
        y_test.append('11')
      else:
        y_test.append(row[0])

x_test = np.array(x_test, str)
x_test = x_test.reshape(16456, 784)
x_test = x_test.astype('float32')
x_test /=  255

y_test = np.array(y_test,int)

print(x_test.shape)

# load the model and create predictions on the test set
predicted_classes = mnist_model.predict_classes(x_test)
predicted_classes = np.array(predicted_classes,int)

print(predicted_classes[10000])
print(y_test[10000])
print(predicted_classes[1234])
print(y_test[1234])
print(predicted_classes[2])
print(y_test[2])
print(predicted_classes[3])
print(y_test[3])

count = 0
for elm in y_test:
  y_test[count] = int(elm)

  count += 1

print(y_test.shape)
print(y_test[1])
print(predicted_classes.shape)
print(predicted_classes)
print(y_test)


# see which we predicted correctly and which not
correct_indices = np.equal(predicted_classes,y_test)
incorrect_indices = np.nonzero(correct_indices)

print(correct_indices)
print(len(correct_indices)," classified correctly")
print(len(incorrect_indices)," classified incorrectly")

# adapt figure size to accomodate 18 subplots
plt.rcParams['figure.figsize'] = (7,14)

figure_evaluation = plt.figure()

# plot 9 correct predictions
for i, correct in enumerate(correct_indices[:11]):
    plt.subplot(6,3,i+1)
    print(correct)
    plt.imshow(x_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title(
      "Predicted: {}, Truth: {}".format(predicted_classes[correct],
                                        y_test[correct]))
    plt.xticks([])
    plt.yticks([])

# plot 9 incorrect predictions
for i, incorrect in enumerate(incorrect_indices[:11]):
    plt.subplot(6,3,i+10)
    plt.imshow(x_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title(
      "Predicted {}, Truth: {}".format(predicted_classes[incorrect],
                                       y_test[incorrect]))
    plt.xticks([])
    plt.yticks([])

figure_evaluation
