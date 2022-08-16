#MNIST를 Tensorflow High Level API로 구현하는 CNN 예제입니다.

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#MNIST 다운로드
mnist      = input_data.read_data_sets('c:/tmp/mnistdata', one_hot=True) #숫자 하나만 선택되도록

tf.set_random_seed(777)  # reproducibility

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

#Training, Testing, Prediction을 Function으로 구현한 Utility Class
class Model:

    #클래스 초기화
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    #네트워크 정의
    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.training = tf.placeholder(tf.bool)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 784])

            # img 28x28x1 (black/white), Input Layer
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3], #필터32개, 필터크기3x3
                                     padding="SAME", activation=tf.nn.relu) #Feature Map 크기변동 없음
            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], #MaxPooling 필터크기 2x2
                                            padding="SAME", strides=2) #크기 SAME이나 Stride=2로 인해 1/2로 축소됨
            dropout1 = tf.layers.dropout(inputs=pool1,
                                         rate=0.7, training=self.training) #drop_out 0.7%, training=True 일때만 dropout 실행

            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3], #필터64개, 필터크기3x3
                                     padding="SAME", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2,
                                         rate=0.7, training=self.training)

            # Convolutional Layer #2 and Pooling Layer #2
            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3], #필터126개, 필터크기3x3
                                     padding="same", activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                            padding="same", strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3,
                                         rate=0.7, training=self.training)

            # Fully Connected Layer
            flat = tf.reshape(dropout3, [-1, 128 * 4 * 4])
            dense4 = tf.layers.dense(inputs=flat,
                                     units=625, activation=tf.nn.relu)
            dropout4 = tf.layers.dropout(inputs=dense4,
                                         rate=0.5, training=self.training)

            # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            self.logits = tf.layers.dense(inputs=dropout4, units=10)

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #모델(logits)로 예측 수행
    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    #정확도 계산
    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    #학습
    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})

with tf.Session() as sess:
    
    m1 = Model(sess, "m1") #객체지향 클래스 선언
    sess.run(tf.global_variables_initializer()) #세션 초기화
    
    # Training 사이클
    for epoch in range(training_epochs): #반복횟수
        avg_cost = 0. #평균 Cost 변수
        
        total_batch = int(mnist.train.num_examples / batch_size) #Loop를 도는 횟수 계산
        
        # Loop 실행
        for i in range(total_batch):
            
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  #mnist.train.next_batch 함수로 batch_size만큼 한번에 꺼냄
            
            # Fit training using batch data
            _cost, _ = m1.train(batch_xs, batch_ys)
            
            # 평균 Cost 계산
            avg_cost += _cost / total_batch
        print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)) #진행상황 출력
        
    print("학습 끝!")
    
    # Test 및 정확드 계산
    print('정확도:', m1.get_accuracy(mnist.test.images, mnist.test.labels))

