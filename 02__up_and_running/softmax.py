import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


DATA_DIR = '/tmp/data'
NUM_STEPS = 1000
MINIBATCH_SIZE = 100


data = input_data.read_data_sets(DATA_DIR, one_hot=True) ## Load the mnist
print ("data: ", type(data.train))

x = tf.placeholder(tf.float32, [None, 784]) # 784 = 28 x 28 pixel
W = tf.Variable(tf.zeros([784, 10]))

y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=y_pred, labels=y_true))

gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1)) # argmax: extract the dimension which is maximum
print ("correct_mask: ",  type(correct_mask))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32)) # casting tf.Tensor to the float32

with tf.Session() as sess:
    # Train
    sess.run(tf.global_variables_initializer())
    for cnt in range(NUM_STEPS):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE) # setting batch normalization
        
        if cnt == 1:
            print (batch_xs.shape, batch_ys.shape) # (100, 784), (100, 10), ndarray
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})

    # Test
    ans = sess.run(accuracy, feed_dict={x: data.test.images, y_true: data.test.labels})

print("Accuracy: {:.4}%".format(ans*100))
