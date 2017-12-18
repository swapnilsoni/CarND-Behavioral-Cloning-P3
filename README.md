# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/pre-processed-image.png "preporcessed image"
[image2]: ./images/fliped-image.png "Flipped Image"
[image3]: ./images/model_architecture.png "Flipped Image"
#### 1. My Submission includes all required files

My project includes the following files:
* [model.py](https://github.com/swapnilsoni/CarND-Behavioral-Cloning-P3/blob/master/model.py) containing the script to create and train the model
* [drive.py](https://github.com/swapnilsoni/CarND-Behavioral-Cloning-P3/blob/master/drive.py) for driving the car in autonomous mode
* [model.h5]()  Could not upload as file size is more than 100MB

#### 2. Pre-processing images
Gray scaled image
![Gray image][image1]
Flipped gray scaled image
![Flipped image][image2]

#### 3. An appropriate model architecture has been employed
1. My model inspired by Invidia [PDF](https://arxiv.org/pdf/1604.07316.pdf)
2. Cropped images is also part of the model
![Gray image][image3] 

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes. Also used RELU to introduce nonlinearity

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 140).

#### 4. Appropriate training data and model.py

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving.

Approach to develop the model:
1) Preprocessed images: Turned color image to YUV and also applied gaussian blur.
2) Fliped the image

Validation:
The data set has been divided into two parts train and valid set. The training data is 80% and validation set 20%.
Epoch: 5
batch size: 128
Loss function: MSE
Shuffle: True

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track when there are shadowes of the trees on the track. To improve the driving behavior in these cases, I need remove those shadowes on preprocessing

#### 5. Video
[![video](https://github.com/swapnilsoni/CarND-Behavioral-Cloning-P3/blob/master/images/video.png)](https://github.com/swapnilsoni/CarND-Behavioral-Cloning-P3/blob/master/video.mp4)
