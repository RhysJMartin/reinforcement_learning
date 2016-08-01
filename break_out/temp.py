import tensorflow as tf

session = tf.Session()
state = tf.placeholder("float", [None, 3])

weights = tf.Variable(tf.constant(0., shape=[3, 2]))

value_function = tf.matmul(state, weights)

session.run(tf.initialize_all_variables())

ans = session.run(value_function, feed_dict={state: [[1., 0., 0.]]})

print(ans)