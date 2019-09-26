import capslayer as cl
import tensorflow as tf

mnist=tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train,x_test=x_train/255.0,x_test/255.0
n_classes=10

x=tf.placeholder(dtype=tf.float32,shape=[-1,28,28,1])
y=tf.placeholder(dtype=tf.int32,shape=[-1,n_classes])

conv_args={
	"filters":256,
	"kernel_size":9,
	"strides":1,
	"padding":"VALID",
	"activation":tf.nn.relu
}
net=tf.layers.conv2d(x,**conv_args)

primary_args={
	"filters":32,
	"kernel_size":9,
	"strides":2,
	"out_caps_dims":[8,1]
}
net,activation=cl.layers.primaryCaps(net,**primary_args)

digit_args={
	"num_outputs":n_classes,
	"out_caps_dims":[16,1],
	"routing_method":"DynamicRouting"
}
pose,prob=cl.layers.dense(net,activation,**digit_args)

margin_loss=cl.losses.margin_loss(y,prob)

