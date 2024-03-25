import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
v = tf.constant([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.])
w = tf.constant([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.])
#2 4 6 8 10
sess = tf.Session()
# print(sess.run(v))
def func(x):
    return tf.exp(-x/40)
def cut(x):
    return 1/(1+tf.exp(-9.0*(x-5.0)))
print(sess.run((tf.exp(-w))))
print(sess.run(cut(v)*func(v)))
sess.close()

