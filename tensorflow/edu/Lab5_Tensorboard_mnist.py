from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#MNIST 다운로드
mnist      = input_data.read_data_sets('/tensorflow/mnistdata', one_hot=True) #숫자 하나만 선택되도록

# Hyper Parameters
learning_rate = 0.05
training_epochs = 20
batch_size = 100
display_step = 1
logs_path = '/tensorflow/log'

# tf Graph Input
x_input = tf.placeholder(tf.float32, [None, 784], name='InputData') # MNIST 이미지 사이즈 28*28=784
y_label = tf.placeholder(tf.float32, [None, 10], name='LabelData')  # 0-9 범위안의 답 => 10 classes

# Weight와 Bias 초기화 
W1 = tf.Variable(tf.random_normal([784, 100], mean=0.0, stddev=1.0))
b1 = tf.Variable(tf.zeros([100]))
W2 = tf.Variable(tf.random_normal([100, 10], mean=0.0, stddev=1.0))
b2 = tf.Variable(tf.zeros([10]))

with tf.name_scope('Model'):
    hidden1 = tf.sigmoid(tf.matmul(x_input, W1) + b1)
    predict = tf.matmul(hidden1, W2) + b2 #분류를 위해 Softmax가 필요하나, 아래 cost함수에서 함께해 줌
    
with tf.name_scope('Loss'):
    cross_entropy_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= y_label, logits= predict))
    
with tf.name_scope('SGD'): # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_cost)
    
with tf.name_scope('Accuracy'): # 정확도
    acc = tf.equal(tf.argmax(predict, 1), tf.argmax(y_label, 1)) #prediction과 label이 같은지 비교
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))
    
tf.summary.histogram("x_input", x_input)
tf.summary.histogram("predict", predict)
tf.summary.histogram("y_label", y_label)
tf.summary.scalar("loss", cross_entropy_cost)
tf.summary.scalar("accuracy", acc)

merged_summary_op = tf.summary.merge_all() #모든 summary 합치기

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Tensorboard에 쓰기
    summary_writer = tf.summary.FileWriter(logs_path, graph=sess.graph)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            _, c, summary = sess.run([optimizer, cross_entropy_cost, merged_summary_op],
                                     feed_dict={x_input: batch_xs, y_label: batch_ys})

            summary_writer.add_summary(summary, epoch * total_batch + i)

            avg_cost += c / total_batch
        accuracy = sess.run(acc, feed_dict={x_input: mnist.test.images, y_label: mnist.test.labels})
        print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost), "정확도:", "{:.9f}".format(accuracy)) #진행상황 출력

    print("학습 끝!")
    
    print("Command line: tensorboard --logdir=",logs_path , "\n브라우저에서 http://localhost:6006/ 에 접속")
