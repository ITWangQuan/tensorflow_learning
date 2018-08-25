import tensorflow as tf
var = tf.Variable(0)
add_operation = tf.add(var, 1)
update_operation = tf.assign(var, add_operation)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for _ in range(3):
        sess.run(update_operation)
        result = sess.run(var)
        print(result)
