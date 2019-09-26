import capslayer as cl
import tensorflow as tf
import os
import numpy as np

batch_size = 50

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
n_classes = 10

x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
y = tf.placeholder(dtype=tf.int32, shape=[None])

conv_args = {
	"filters": 256,
	"kernel_size": 9,
	"strides": 1,
	"padding": "VALID",
	"activation": tf.nn.relu
}
net = tf.layers.conv2d(x, **conv_args)

primary_args = {
	"filters": 32,
	"kernel_size": 9,
	"strides": 2,
	"out_caps_dims": [8, 1]
}
net, activation = cl.layers.primaryCaps(net, **primary_args)

num_caps = np.prod(cl.shape(net)[1:4])
net = tf.reshape(net, shape=[-1, num_caps, 8, 1])
activation = tf.reshape(activation, shape=[-1, num_caps])

digit_args = {
	"num_outputs": n_classes,
	"out_caps_dims": [16, 1],
	"routing_method": "DynamicRouting"
}
pose, prob = cl.layers.dense(net, activation, **digit_args)

T=tf.one_hot(y,depth=10)
margin_loss = cl.losses.margin_loss(T, prob)

cost = tf.reduce_mean(margin_loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prob, 1), tf.argmax(T, 1)), "float"))
saver = tf.train.Saver
optimizer = tf.train.AdamOptimizer().minimize(cost)
init = tf.global_variables_initializer()

if not os.path.exists("./saved_model"):
	os.mkdir("./saved_model")
save_path = "./saved_model/saved_model.ckpt"

restore = False

with tf.Session() as sess:
	if restore:
		saver.restore(sess, save_path)
		print("==============")
		print("model restored")
		print("==============")
	else:
		sess.run(init)
		print("==============")
		print("initialized")
		print("==============")
	
	for iter in range(3000):
		idx = np.random.choice(x_train.shape[0], size=batch_size, replace=False)
		train_x = x_train[idx, :]
		train_y = y_train[idx]
		_, batch_cost, batch_acc = sess.run([optimizer, cost, accuracy], feed_dict={x: train_x.reshape([-1, 28, 28, 1]),
																					y: train_y})
		if iter % 100 == 99:
			print("iter:%5d, cost:%.5f, acc:%.4f" % (iter + 1, batch_cost, batch_acc))
		if iter % 1000 == 999:
			idx = np.random.choice(x_test.shape[0], size=batch_size, replace=False)
			test_x = x_test[idx, :]
			test_y = y_test[idx]
			print("test batch accuracy:%.4f" % sess.run(accuracy, feed_dict={x: test_x.reshape([-1, 28, 28, 1]),
																			 y: test_y}))
			# saver.save(sess, save_path)
	arr=sess.run(prob,feed_dict={x:x_test[0:10].reshape([-1, 28, 28, 1]),y:y_test[0:10]})
	for item in arr:
		for i in item:
			print("%.6f"%i,end=" ")
		print()
	print("FINISHED")
