from __future__ import print_function #Python2.x에서도 print문 실행되도록 하기 위해 추가
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#MNIST 다운로드
mnist      = input_data.read_data_sets('/tensorflow/mnistdata', one_hot=True) #숫자 하나만 선택되도록

# tf Graph Input
x_input = tf.placeholder(tf.float32, [None, 784]) # MNIST 이미지 사이즈 28*28=784
y_label = tf.placeholder(tf.float32, [None, 10]) # 0-9 범위안의 답 => 10 classes

# Weight 정규분포로 초기화 
W1 = tf.Variable(tf.random_normal([784, 100], mean=0.0, stddev=1.0))
b1 = tf.Variable(tf.zeros([100]))
W2 = tf.Variable(tf.random_normal([100, 10], mean=0.0, stddev=1.0))
b2 = tf.Variable(tf.zeros([10]))

# Weight와 Bias o으로 초기화 실험
#W1 = tf.Variable(tf.zeros([784, 100]))
#b1 = tf.Variable(tf.zeros([100]))
#W2 = tf.Variable(tf.zeros([100, 10]))
#b2 = tf.Variable(tf.zeros([10]))

# Model 정의
hidden1 = tf.sigmoid(tf.matmul(x_input, W1) + b1)
y_predict = tf.matmul(hidden1, W2) + b2 #분류를 위해 Softmax가 필요하나, 아래 cost함수에서 함께해 줌

# Error Cost 계산
cross_entropy_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= y_label, logits= y_predict))

# Gradient Descent Optimizer (learning_rate = 0.1)
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_cost)

# Accuracy 계산
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_label, 1)) #prediction과 label이 같은지 비교
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batch_size = 100 #한번에 가져오는 데이터 수
# 변수 초기화
init = tf.global_variables_initializer()

# Graph 생성
with tf.Session() as sess:
    sess.run(init)

    # Training 사이클
    for epoch in range(60): #반복횟수
        avg_cost = 0. #평균 Cost 변수
        
        total_batch = int(mnist.train.num_examples / batch_size) #Loop를 도는 횟수 계산
        
        # Loop 실행
        for i in range(total_batch):
            
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  #mnist.train.next_batch 함수로 batch_size만큼 한번에 꺼냄
            
            # Fit training using batch data
            _, cost = sess.run([optimizer, cross_entropy_cost], feed_dict={x_input: batch_xs, y_label: batch_ys})
            
            # 평균 Cost 계산
            avg_cost += cost / total_batch
        acc = sess.run(accuracy, feed_dict={x_input: mnist.test.images, y_label: mnist.test.labels})
        print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost), "정확도:", "{:.9f}".format(acc)) #진행상황 출력
        
    print("학습 끝!")
