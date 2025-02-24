import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# 加载训练好的模型（假设模型已保存）
model = tf.keras.models.load_model('mnist_model.h5')

# 加载并预处理本地图片
img_path = '9.jpg'
img = image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# 预测
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)

# 可视化图片和预测结果
plt.imshow(img_array[0], cmap='gray')
plt.title(f"预测的数字: {predicted_class}")
plt.show()

# 输出预测
print(f"预测的数字是: {predicted_class}")
