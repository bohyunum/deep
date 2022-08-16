from __future__ import print_function #Python2.x에서도 print문 실행되도록 하기 위해 추가
import tensorflow as tf

a = tf.constant(5) #input a
b = tf.constant(3) #input b
c = tf.multiply(a, b)   #a 곱하기 b
d = tf.add(a, b)   #a 더하기 b
e = tf.add(c, d)   #c 더하기 d

print(c)
print(d)
print(e)

sess = tf.Session() #세션 선언

print(sess.run(c))
print(sess.run(d))
print(sess.run(e))

sess.close() #세션닫기





