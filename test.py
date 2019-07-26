import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
batch_size=100

mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
n_batch=mnist.train.num_examples//batch_size
#定义一个添加层的函数
def add_layer(input,insize,outsize,activation_function=None):
    w=tf.Variable(tf.random_normal([insize,outsize]))
    b=tf.Variable(tf.random_normal([outsize]))
    w_mul_x_plus_b=tf.add(tf.matmul(input,w),b)
    if activation_function==None:
        output=w_mul_x_plus_b
    else:
        output=tf.nn.relu(w_mul_x_plus_b)
    return output
xs=tf.placeholder(tf.float32,[None,784])
ys=tf.placeholder(tf.float32,[None,10])

y=add_layer(xs,784,10,activation_function=tf.nn.softmax)

#定义损失函数
loss=tf.reduce_mean(tf.square(y-ys))
#定义训练过程
train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)
correct_prediction=tf.equal(tf.argmax(ys,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#初始化所有的变量
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    train_step=21
    for i in range(train_step):
        #将每个图片都训练一次
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(loss,feed_dict={xs:batch_xs,ys:batch_ys})
        acc=sess.run(accuracy,feed_dict={xs:mnist.test.images,ys:mnist.test.labels})
        print("Iter"+str(i)+",accuracy:"+str(acc))

    