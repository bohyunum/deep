from __future__ import print_function #Python2.x에서도 print문 실행되도록 하기 위해 추가
import tensorflow as tf

input1 = tf.constant([3.,3.,5.,6.], shape=[1, 4]) #input 1 x 4

W1 = tf.Variable(tf.zeros([4, 2])) #hidden1 4 x 2. 0으로 초기화(평균:0 표준편차:1.0)
b1 = tf.Variable(tf.zeros([2])) #hidden1 bias

W2 = tf.Variable(tf.zeros([2, 3])) #hidden2 2 x 3
b2 = tf.Variable(tf.zeros([3])) #hidden2 bias

W3 = tf.Variable(tf.zeros([3, 1])) #output 3 x 1
b3 = tf.Variable(tf.zeros([1])) #output bias

sum1 = tf.matmul(input1, W1) + b1  #Summary
out1 = tf.nn.sigmoid(sum1)  #Activation Function

sum2 = tf.matmul(out1, W2) + b2
out2 = tf.nn.sigmoid(sum2)

final = tf.matmul(out2, W3) + b3 #Output은 Softmax 또는 없이

init = tf.global_variables_initializer() #tensor초기화

with tf.Session() as sess:
    sess.run(init) #초기화 실행

    print("sum1= ",sess.run(sum1));
    print("out1= ",sess.run(out1));
    print("sum2= ",sess.run(sum2));
    print("out2= ",sess.run(out2));
    print("final= ",sess.run(final));
