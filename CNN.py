import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
batch_size=100
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
#计算总共有多少批
n_batch=mnist.train.num_examples//batch_size
#定义卷积神经网络中需要用到的参数
def weight_variable(shape):
    #截断正态分布指的是限制变量取值范围的一种分布，而正态分布表示不进行任何截断的分布
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
#定义卷积神经网络中的卷积操作和池化操作
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="SAME")
def max_pool_2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#定义输入和输出的占位符
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])

#将输入转换成为4D向量形式，以符合卷积神经网络的输入
x_img=tf.reshape(x,[-1,28,28,1])
#定义每一个卷积神经网络中的参数和权重
#第一个卷积层的参数
w_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])

#输入和第一个卷积层进行卷积，然后进行池化操作
h_conv1=tf.nn.relu(conv2d(x_img,w_conv1)+b_conv1)
h_pool1=max_pool_2(h_conv1)
#此时经过第一个卷积层之后参数变为14*14
#第二个卷积层的参数
w_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
#第一个卷积层池化的输出与第二层进行卷积
h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2=max_pool_2(h_conv2)
#此时第二个池化层的输出为7*7
#后面接一个全连接层，因此要将池化层的输出拉平
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
#定义全连接层的权重和偏置
w_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])

#计算第一个全连接层的输出
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)
#添加dropout,其中keep_drop表示每个元素被保留下来的概率，也就是神经元被选中的概率
#给keep_drop定义占位符，存放该值，在run的时设置其具体值
keep_drop=tf.placeholder(tf.float32)
#第一个全连接层dropout之后的输出
h_fc1_drop=tf.nn.dropout(h_fc1,keep_drop)
#定义第二个全连接层的权重和参数
w_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
#计算第二个全连接层的输出
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

#定义损失函数
#logits参数就是神经网络最后一层的输出，大小[batch_size,num_classes],labels参数是实际的标签
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#定义训练过程，使用AdamOptimizer进行优化
train_step=tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#初始化所有的变量
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_drop:0.7})
        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_drop:1.0})
        print("Iter"+str(i)+',Test accuracy'+str(acc))









