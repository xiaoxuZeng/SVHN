import tensorflow as tf
from scipy.io import loadmat as load
import numpy

#load train_data
"""包括预处理部分，主要是将标签为10的数据给滤出，并将数据进行一些转换，使其符合输入格式。
训练数据和测试数据都来自于SVHN官网format2"""
train_data =load('/home/zzy/PycharmProjects/PJ1_camera/train_32x32.mat')

x = train_data['X']
y = train_data['y']
x1 = numpy.zeros((68310,32,32,3),dtype='float16')
y_1 = numpy.zeros((68310,10),dtype='float16')
y1 = numpy.zeros((68310,1),dtype='int8')
j = 0
for i in range(73257):
    if y[i] != 10:
        j=j+1
        x1[j] = x[:,:,:,i]
        y1[j] = y[i]
        y_1[j] = numpy.eye(10)[y1[j]]

#load test_data
train_data =load('/home/zzy/PycharmProjects/PJ1_camera/test_32x32.mat')

x2 = train_data['X']
y2 = train_data['y']
x3 = numpy.zeros((24289,32,32,3),dtype='float16')
y_2 = numpy.zeros((24289,10),dtype='float16')
y3 = numpy.zeros((24289,1),dtype='int8')
for i in range(26032):
    if y2[i] != 10:
        j=j+1
        x3[j] = x2[:,:,:,i]
        y3[j] = y2[i]
        y_2[j] = numpy.eye(10)[y3[j]]


#创建一个交互式Session
sess = tf.InteractiveSession()

#创建两个占位符，x为输入网络的图像，y_为输入网络的图像类别
x = tf.placeholder("float", shape=[None,32,32,3])
y_ = tf.placeholder("float", shape=[None, 10])

# 定义variable_summaries
def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

#权重初始化函数
def weight_variable(shape):
    #输出服从截尾正态分布的随机值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#偏置初始化函数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#创建卷积op
#x 是一个4维张量，shape为[batch,height,width,channels]
#卷积核移动步长为1。填充类型为SAME,可以不丢弃任何像素点
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

#创建池化op
#采用最大池化，也就是取窗口中的最大值作为结果
#x 是一个4维张量，shape为[batch,height,width,channels]
#ksize表示pool窗口大小为2x2,也就是高2，宽2
#strides，表示在height和width维度上的步长都为2
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],
                          strides=[1,2,2,1], padding="SAME")


#第1层，卷积层
#初始化W为[1,1,3,32]的张量，表示卷积核大小为5*5，第一层网络的输入和输出神经元个数分别为1和32
W_conv1 = weight_variable([1,1,3,32])

#初始化b为[32],即输出大小
b_conv1 = bias_variable([32])

x_image = x

tf.summary.image(
    name='x_image',
    tensor=x_image,
    max_outputs=3,
    collections=None,
    family=None
)

#把x_image和权重进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max_pooling
#h_pool1的输出即为第一层网络输出，shape为[batch,16,16,3]
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第2层，卷积层
#卷积核大小是2*2，这层的输入和输出神经元个数为32和64
W_conv2 = weight_variable([2,2,32,64])
b_conv2 = weight_variable([64])

#tf.summary.histogram('W_conv2',W_conv2)
#tf.summary.histogram('b_conv2',b_conv2)

#h_pool2即为第二层网络输出，shape为[batch,8,8,1]
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#第3层，卷积层
#卷积核大小是3*3，这层的输入和输出深度为64和256
W_conv3 = weight_variable([3,3,64,256])
b_conv3 = weight_variable([256])

#tf.summary.histogram('W_conv3',W_conv3)
#tf.summary.histogram('b_conv3',b_conv3)

#h_pool2即为第二层网络输出，shape为[batch,8,8,256]
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

#第4层, 全连接层
#这层是拥有1024个神经元的全连接层
#W的第1维size为7*7*64，7*7是h_pool2输出的size，64是第2层输出神经元个数
W_fc1 = weight_variable([4*4*256, 2048])
b_fc1 = bias_variable([2048])

#tf.summary.histogram('W_fc1',W_fc1)
#tf.summary.histogram('b_fc1',b_fc1)

#计算前需要把第2层的输出reshape成[batch, 7*7*64]的张量
h_pool2_flat = tf.reshape(h_pool3, [-1, 4*4*256])
h_fc1 = tf.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout层
#为了减少过拟合，在输出层前加入dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#输出层
#最后，添加一个softmax层
#可以理解为另一个全连接层，只不过输出时使用softmax将网络输出值转换成了概率
W_fc2 = weight_variable([2048, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


#预测值和真实值之间的交叉墒
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))


tf.summary.scalar('cross_entropy', cross_entropy)


#train op, 使用ADAM优化器来做梯度下降。学习率为0.0001
train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)

#评估模型，tf.argmax能给出某个tensor对象在某一维上数据最大值的索引。
#因为标签是由0,1组成了one-hot vector，返回的索引就是数值为1的位置
correct_predict = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

tf.summary.scalar('y_conv', tf.reduce_mean(tf.argmax(y_conv, 1)))
tf.summary.scalar('y_', tf.reduce_mean(tf.argmax(y_, 1)))
tf.summary.histogram('y_conv1',tf.argmax(y_conv, 1))
tf.summary.histogram('y_1',tf.argmax(y_, 1))

#计算正确预测项的比例，因为tf.equal返回的是布尔值，
#使用tf.cast把布尔值转换成浮点数，然后用tf.reduce_mean求平均值
accuracy = tf.reduce_mean(tf.cast(correct_predict, "float"))

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('/home/zzy/PycharmProjects/PJ1_camera', sess.graph)


#初始化变量
sess.run(tf.initialize_all_variables())


"""save model"""

saver = tf.train.Saver()
model_path = "/home/zzy/PycharmProjects/PJ1_camera/savedmodel/cnn"


"""用于区分训练还是测试"""
flag_train = False

"""开始训练模型，循环200000次，每次随机从训练集中抓取40幅图像"""
if flag_train == True:
    batch = x1
    j = 0
    for i in range(250000):
        j= j+1
        if i%100==0:
            #每100次输出一次日志
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[40*j:40*j+40], y_:y_1[40*j:40*j+40],keep_prob: 1})
            print("step %d, training accuracy %g" % (i, train_accuracy))
            summary, _ = sess.run([merged, train_step], feed_dict={
                x: batch[40 * j:40 * j + 40], y_: y_1[40 * j:40 * j + 40],keep_prob: 1})
            train_writer.add_summary(summary, i)
            save_path = saver.save(sess, model_path)
        if j == 1700:
            j=0
        train_step.run(feed_dict={x:batch[40*j:40*j+40], y_:y_1[40*j:40*j+40],keep_prob: 1})

"""进行测试"""
if flag_train == False:
    saver.restore(sess,model_path)
    batch1 = x3
    print('test accuracu %g' %accuracy.eval(feed_dict={x:batch1[0:7500], y_:y_2[0:7500], keep_prob: 1}))
