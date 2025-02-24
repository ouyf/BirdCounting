from ultralytics import YOLO
import torch


if __name__ == "__main__":
# 加载预训练模型
	model = YOLO('yolo11n.pt')  # 'yolov8n.pt' 是 YOLOv8 的轻量级版本

# Train the model using the 'coco8.yaml' dataset for 3 epochs
	results = model.train(data=".\\birds-Train.yaml", epochs=600, imgsz=640)

	# Evaluate the model's performance on the validation set
	results = model.val()

	#torch.save(model.state_dict(), 'birds_weights.pt')
	#model.export()

	# 使用模型进行推理
	results = model('.\\test10.jpg')  # 输入图像路径

	num_objects = len(results[0].boxes)

	print(f"Detected {num_objects} objects.")

	# 查看检测结果
	results[0].show()  # 显示图像和检测框


