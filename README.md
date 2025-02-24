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

2. VisionAgent
   
   VisionAgent generate Visual AI code from prompts. VisionAgent selects models for your vision tasks, so you can build vision-enabled apps in minutes: https://landing.ai/visionagent
   
   So we use it to generate visual AI code to count the birds number of the image, the result looks acceptable:
   
   image: dcuks  number:33
   
   ![](https://github.com/ouyf/BirdCounting/blob/main/docs/img/ducks.png)
   
   image: flying-birds number: 171
   
   ![](https://github.com/ouyf/BirdCounting/blob/main/docs/img/flying-birds.png)
   
   But there are still some counting errors that can be improved.
   
   The current vision ai code are visionAgent\birdCount_v1.py, birdCount_v2.py
   
   

3. YOLO






