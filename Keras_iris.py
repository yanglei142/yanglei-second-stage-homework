import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import sklearn.datasets as datasets

# 1. 层定义
# 输入层
x = keras.Input(shape=(4,))  # 输入是4个特征，杨样本数不确定

# 全链接层
layer_1 = keras.layers.Dense(
    units=1,
    activation=keras.activations.sigmoid
)   # 输出的值要么0，要么1


# 2. 定义模型
model = keras.Sequential(layers=[x, layer_1])
# 3. 定义训练参数
model.compile(
    optimizer='adam',     # 指定优化器
    loss='binary_crossentropy',   # 指定损失函数
#     metrics=['accuracy']
)
# 4. 开始训练
# 加载数据
data, target = datasets.load_iris(return_X_y=True)
data = data[:100, :]
target = target[:100]
# 训练
model.fit(x=data,     # 训练样本
          y=target,   # 训练标签
          epochs=2000,
          batch_size=10,
          verbose=0)    # 控制训练过程中的输出
# 5.模型评估
score = model.evaluate(
    x=data,
    y=target)
print(score)
# 6.预测
pre_result = model.predict(data)
category = [0 if item<=0.5 else 1 for item in pre_result ]

accury = (target == category).mean()
print('分类准确度：',accury *100,'%')