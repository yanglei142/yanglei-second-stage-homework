# coding=utf-8
#% matplotlib inline
#import matplotlib.pyplot as plt

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt

# 定义神经网络模型=================
# 定义输入
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])


# 定义训练系数
# 定义系数模板
def get_weights(shape):
    w_init = tf.random.truncated_normal(shape=shape, mean=0, stddev=0.1, dtype=tf.float32)       # 给被训练的变量一个初始值
    b_init = tf.random.truncated_normal(shape=[shape[-1]], mean=0, stddev=0.1, dtype=tf.float32)

    w = tf.Variable(initial_value=w_init)
    b = tf.Variable(initial_value=b_init)
    return w, b


# 定义运算
def layer(in_x, in_w, in_b, padding='VALID'):
    o = tf.nn.conv2d(input=in_x, filter=in_w, strides=[1, 1, 1, 1], padding=padding)
    o = tf.math.add(o, in_b)
    o = tf.nn.relu(o)
    o = tf.nn.avg_pool(value=o, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return o


# 卷积1
w1, b1 = get_weights(shape=[5, 5, 1, 6])
o1 = tf.nn.conv2d(input=x, filter=w1, strides=[1, 1, 1, 1], padding='SAME')
o1 = tf.nn.bias_add(o1, b1)
o1 = tf.nn.relu(o1)
o1 = tf.nn.avg_pool(value=o1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 卷积2
w2, b2 = get_weights(shape=[5, 5, 6, 16])    # 这里的深度6就是上面输出的6
o2 = tf.nn.conv2d(input=o1, filter=w2, strides=[1, 1, 1, 1], padding='VALID')
o2 = tf.nn.bias_add(o2, b2)
o2 = tf.nn.relu(o2)
o2 = tf.nn.avg_pool(value=o2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 卷积3
w3, b3 = get_weights(shape=[5, 5, 16, 120])    # 这里的深度6就是上面输出的6
o3 = tf.nn.conv2d(input=o2, filter=w3, strides=[1, 1, 1, 1], padding='VALID')
o3 = tf.nn.bias_add(o3, b3)
o3 = tf.nn.relu(o3)

# 格式化，进入全连接层
o3 = tf.reshape(o3, [-1, 120])
w4, b4 = get_weights(shape=[120, 84])    # 这里的深度6就是上面输出的6
o4 = tf.nn.relu(tf.matmul(o3, w4) + b4)
o4 = tf.nn.dropout(o4, 0.75)

w5, b5 = get_weights(shape=[84, 10])    # 这里的深度6就是上面输出的6
o5 = tf.nn.softmax(tf.matmul(o4, w5) + b5)

y_ = o5
# 定义损失函数
loss = tf.losses.sigmoid_cross_entropy(y, y_)
# loss = tf.losses.mean_squared_error(y, y_)
# 定义训练算法
# optimizer = tf.train.GradientDescentOptimizer(0.0001)
optimizer = tf.train.AdamOptimizer(0.0001)
trainer = optimizer.minimize(loss)

# 输入数据加载，格式化与规范处理
result = np.loadtxt("train.txt", np.int)
result = result[0:1000]
print("样本个数：(%d)" % len(result))
labels = np.zeros((len(result), 10), dtype=np.int)
for i in range(len(result)):
    lb = result[i]
    labels[i][lb] = 1
# 把图像数据转换成需要的格式
data = np.zeros((len(result), 28, 28, 1), np.float32)
for i in range(len(result)):
    # print('加载图像：%d' % i )
    img = np.array([plt.imread("train/TrainImage_%05d.bmp" % (i+1))])
    data[i, :, :, 0] = img

print("数据加载完毕！")
# 训练
# 定义评估模型
correct = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))


# 初始化
session = tf.Session()
global_v = tf.global_variables()
op_init = tf.initializers.variables(global_v)
session.run(op_init)

TIMES = 500
batch_size = 100
batch = len(data) // batch_size
print(batch)
correct_rates =[]
for t in range(TIMES):
    loss_result = 0.0
    for idx in range(batch):
        _, loss_result = session.run([trainer, loss],
                                     feed_dict={
                                         x: data[idx * batch_size:(idx+1) * batch_size],
                                         y: labels[idx * batch_size:(idx+1) * batch_size]})
    # 没一轮训练就评估效果
    if t % 5 == 0:
        correct_rate = session.run(accuracy, feed_dict={x: data, y: labels})
        print('正确率: %5.2f%%，损失度：%f' % (correct_rate * 100.0, loss_result))
        correct_rates.append(correct_rate)
print('训练完毕')
# 可视化一下训练过程
figure = plt.figure(figsize=(8,4))
ax = figure.add_axes([0.1,0.1,0.8,0.8])
ax.plot(range(len(correct_rates)),correct_rates, color=(0,0,1,1), marker='.', label='正确率曲线',
         markerfacecolor=(1,0,0,1),markeredgecolor=(1,0,0,1), markersize=3)
ax.set_xbound(lower=-1, upper=len(correct_rates))
ax.set_ybound(lower=0, upper=1)
plt.annotate(s='最高识别率：%5.2f%%' % (max(correct_rates)*100.0), xy=(60,0.5))
plt.legend()
# plt.grid(b=True)

plt.show()
# 预测