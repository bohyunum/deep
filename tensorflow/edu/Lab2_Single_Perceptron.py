from __future__ import print_function #Python2.x에서도 print문 실행되도록 하기 위해 추가
import tensorflow as tf

x = tf.constant([[2.,3.,-1.,5.,1.]]) # 1x5 matrix
W = tf.constant([[0.1],[0.5],[2.5],[0.2],[3.0]]) #5x1 matrix

model = tf.matmul(x, W)
output = tf.nn.sigmoid(model)

with tf.Session() as sess: #sess.close() 안해도 됨
    writer = tf.summary.FileWriter("./board/LinearRegression", sess.graph)

    print("sum=",sess.run(model));
    print("sigmoid=",sess.run(output));



