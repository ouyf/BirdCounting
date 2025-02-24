import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 1. 加载 MNIST 数据集（手写数字数据集）
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 归一化数据，将像素值从 [0, 255] 缩放到 [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. 可视化前 5 张训练图片及其标签
for i in range(5):
    plt.imshow(x_train[i], cmap='gray')  # 使用灰度图显示
    plt.title(f"标签: {y_train[i]}")
    plt.show()

# 4. 构建一个简单的神经网络模型
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # 将 28x28 图像展平成一维数组
    keras.layers.Dense(1024, activation='relu'),  # 隐藏层：1024 个神经元，ReLU 激活函数
    keras.layers.Dense(512, activation='relu'),   # 新增的隐藏层（512 个神经元）
    keras.layers.Dense(10, activation='softmax') # 输出层：10 个类别，Softmax 归一化
])

# 5. 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6. 训练模型
model.fit(x_train, y_train, epochs=5)

# 7. 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc}")

# 8. 保存模型
model.save('mnist_model.h5')
print("\n模型已保存为 'mnist_model.h5'")
