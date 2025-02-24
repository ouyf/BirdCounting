# BirdCounting

Use AI models to count the birds number in one image or video.

# Background

In bird watching, identifying the species and type of a bird is the most important task. Many AI models already assist with this, such as *Merlin Bird ID*, *Audubon Bird Guide*, and *DongNiao*, among others.

However, another crucial piece of data—the number of birds—often gets overlooked, especially when submitting birdwatching records.

Counting is easy when there are only a few birds for humankind, but it becomes challenging when there’s a flock. This issue is particularly common in bird migration monitoring, where people often resort to estimating the number of birds in a flock. Unfortunately, these estimates are frequently inaccurate and can be far from the actual count.

# Chanllenges

Unlike human counting, using AI (specifically Neural Networks) to count the number of birds must address the following challenges:

1. **Detecting and identifying the birds in images or videos.** Birds have different shapes when flying and when standing, and there is a wide variety in size—from small to large. Despite these differences, the basic characteristics of birds remain the same: one bill, two wings, and two legs.
2. **Handling complex foregrounds or backgrounds.** While some images or videos may feature clear blue skies, many are taken in wooded areas, grasslands, or other environments with cluttered backgrounds. These can include branches, buildings, and various other objects. The AI needs to accurately distinguish birds from these surrounding elements.
3. **Dealing with occlusion between birds.** Birds often gather in groups, leading to partial overlap or cover between them. This occlusion is inevitable and can easily result in errors when counting the number of birds in an image or video.

# Solutions

1. **Basic Example in TensorFlow**
   
   TensorFlow provides basic examples for object recognition, such as handwritten digits and general object recognition (e.g., CIFAR-10, which includes categories like cars, cats, birds, etc.). We adjust the image size and some of the CNN parameters, but the model can only detect whether the image is of a bird or something else, with an unsatisfactory accuracy rate.

2. **VisionAgent**
   
   VisionAgent generate Visual AI code from prompts. VisionAgent selects models for your vision tasks, so you can build vision-enabled apps in minutes: https://landing.ai/visionagent
   
   So we use it to generate visual AI code to count the birds number of the image, the result looks acceptable:
   
   image: dcuks  number:33
   
   ![](https://github.com/ouyf/BirdCounting/blob/main/docs/img/visionAgent/ducks.png)
   
   image: flying-birds number: 171
   
   ![](https://github.com/ouyf/BirdCounting/blob/main/docs/img/visionAgent/flying-birds.png)
   
   But there are still some counting errors that can be improved.
   
   The current vision ai code are visionAgent\birdCount_v1.py, birdCount_v2.py
   
   The process of visonAgent AI code :
   
       1. Load the image using load_image.
       2. Subdivide the image into four overlapping sections.
       3. Perform detection on each section using the prompt 'bird, duck'.
       4. Merge bounding boxes from all sections to remove duplicates.
       5. Overlay the bounding boxes and save the resulting image.
       6. Return the total number of detected birds as the final solution.

3. **YOLO**
   
   Ultralytics YOLO is the latest advancement in the acclaimed YOLO (You Only Look Once) series for real-time object detection and image segmentation: https://docs.ultralytics.com/
   
   It provide some pre-trained models but the detetion is not too good on the birds, especially for the birds which wings have black-white colors.
   
   Result of YOLO pretrained model:
   
   ![](https://github.com/ouyf/BirdCounting/blob/main/docs/img/YOLO/yolo-PredefinedResult.png)
   
   ## Training with YOLO
   
   We need to create our own dataset for YOLO training:
   
   1. Use labelImg to create YOLO supported labeling image data. Save the image and labeling files to the corresponding dataset directory.
   
   2. Create training yaml file: birds-Train.yaml.
   
   3. Run the YOLO model and training for 600 epochs. Note: lower or higher epochs, such as 100, 1000, do not have good results.
   
   4. Load the trained YOLO model to test the images.
   
   Result of YOLO especially trained model:
   
   ![](https://github.com/ouyf/BirdCounting/blob/main/docs/img/YOLO/yolo-espTrainedResult.png)
   
   We can see the result is much better now.






