from __future__ import print_function 
import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data

#MNIST 다운로드
mnist      = input_data.read_data_sets('c:/tmp/mnistdata', one_hot=True) #숫자 하나만 선택되도록

# Hyper Parameters
input_size=784
layer1_size = 1024
layer2_size = 1024
output_size = 10
learning_rate = 0.01
training_epochs = 30
batch_size = 100
regularization_rate = 0.001

# tf Graph Input
x_input = tf.placeholder(tf.float32, [None, input_size]) # MNIST 이미지 사이즈 28*28=784
y_label = tf.placeholder(tf.float32, [None, output_size]) # 0-9 범위안의 답 => 10 classes

with tf.variable_scope('scope' + str(random.random())): #get_variable 재실행 시 오류 회피
    # Weight 초기화 
    W1 = tf.get_variable('W1',shape=[input_size, layer1_size],initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.zeros([layer1_size]))
    W2 = tf.get_variable('W2',shape=[layer1_size, layer2_size],initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.zeros([layer2_size]))
    W3 = tf.get_variable('W3',shape=[layer2_size, output_size],initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.zeros([output_size]))

# Model 정의
layer1 = tf.add(tf.matmul(x_input, W1), b1)
layer1 = tf.nn.relu(layer1)

layer2 = tf.add(tf.matmul(layer1, W2), b2)
layer2 = tf.nn.relu(layer2)

y_predict = tf.add(tf.matmul(layer2, W3), b3) #분류를 위해 Softmax가 필요하나, 아래 cost함수에서 함께해 줌

# Error Cost 계산
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= y_label, logits= y_predict))

#Weight Decay 적용
regularizers = tf.nn.l2_loss(W1)  + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)
cost = tf.reduce_mean(cost + (regularization_rate * regularizers))

# Optimizer 
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Accuracy 계산
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_label, 1)) #prediction과 label이 같은지 비교
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # Training 사이클
    for epoch in range(training_epochs): #반복횟수
        avg_cost = 0. #평균 Cost 변수
        
        total_batch = int(mnist.train.num_examples / batch_size) #Loop를 도는 횟수 계산
        
        # Loop 실행
        for i in range(total_batch):
            
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  #mnist.train.next_batch 함수로 batch_size만큼 한번에 꺼냄
            
            # Fit training using batch data
            _, _cost = sess.run([optimizer, cost]
                                         , feed_dict={x_input: batch_xs, y_label: batch_ys})
            
            # 평균 Cost 계산
            avg_cost += _cost / total_batch
        acc = sess.run(accuracy, feed_dict={x_input: mnist.test.images, y_label: mnist.test.labels})
        print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost), "정확도:", "{:.9f}".format(acc)) #진행상황 출력
        
    print("학습 끝!")
