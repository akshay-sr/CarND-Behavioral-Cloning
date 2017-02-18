Table of Contents
=================

   * [Behavioral Cloning](#behavioral-cloning)
   * [Video of the car driving around the track](#Video of the car driving around the track)
   * [Goal](#goal)
   * [Files](#files)
   * [Code Quality](#code-quality)
      * [Functional Code](#functional-code)
      * [Comments inline with code](#comments-inline-with-code)
   * [Model Architecture &amp; Solution Design](#model-architecture--solution-design)
      * [Architecture: nVidia End-to-End Deep Learning Network](#architecture-nvidia-end-to-end-deep-learning-network)
      * [Objective, Loss function and Hyper-Parameter tuning](#objective-loss-function-and-hyper-parameter-tuning)
      * [Controlling Overfitting](#controlling-overfitting)
      * [Image Preprocessing](#image-preprocessing)
      * [Steering Angle Preprocessing](#steering-angle-preprocessing)
   * [Training Strategy](#training-strategy)
      * [Building an initial Model with data with key insgihts ](#building-an-overfitted-model-with-key-insights)
      * [Data augmentation for balancing the set](#data-augmentation-for-balancing-the-set)
   * [Acknowledgements &amp; References](#acknowledgements--references)

---

# Behavioral Cloning
Using Keras to make building deep neural networks for predicting autonomous steering angles.

# Video of the car driving around the track
This video shows the car driving around the track for the entire lap.

---
# Goal

The goals of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

# Files
My project includes the following files:
* `model.py` - the main script to create and train the model
* `drive.py` - the script making steering angle predictions and feeding it to the simulator, thus enabling driving the car in autonomous mode. 
* `model.h5` - a trained convolution neural network with weights.
* `README.md` - description of the development process (this file)
* Udacity Dataset - Track1 Dataset Used for Training. [Download here](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)
* Unity3D Simulator - [Github Repo](https://github.com/udacity/self-driving-car-sim).

Repository includes all required files and can be used to run the simulator in autonomous mode.

# Code Quality
## Functional Code
Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing
```
$ python drive.py model.h5
```

## Comments inline with code

The `model.py` file contains the entire code with appropriate comments for training and saving the CNN design. It contains comments to explain how the code works.

# Model Architecture & Solution Design

## Architecture: nVidia End-to-End Deep Learning Network

My model consists of a convolution neural network with 3 5x5 filter sizes of depths 24, 36, 48.
Followed by 2 3x3 filtes of depth 64 and 64. And then finally flattening the output and reducing the dimensionality via fully-connected dense layers to a steering angle prediction.
Details of padding and sampling for each convolution layer is specified with detailed comments in code. Reiterating them here

1.Set the first layer to a Convolution2D layer with 5x5 kernel, input_shape set to (66, 200, 3) and subsample to (1,1)
2.Use a MaxPooling2D layer that subsamples by (2,2) after the previous convolution.
3.Use a Dropout layer at 0.5 dropout, following the pooling layer.
4.Use a ReLU activation function after the Dropout layer.
5.Set the fifth layer to a Convolution2D layer with 5x5 kernel, valid padding and subsample set to (2,1). The input shape from the previous operations result in (w,h)=(31,98)
6.Use a MaxPooling2D layer that subsamples by (1,2) after the previous convolution in the fifth layer.
7.Use a Dropout layer at 0.5 dropout, following the pooling layer.
8.Use a ReLU activation function after the Dropout layer.
9.Set the ninth layer to a Convolution2D layer with 5x5 kernel, valid padding and subsample set to (1,2). The input shape from the previous operations result in (w,h)=(14,47)
10.Use a MaxPooling2D layer that subsamples by (2,1) after the previous convolution.
11.Use a Dropout layer at 0.5 dropout, following the pooling layer.
12.Use a ReLU activation function after the Dropout layer.
13.Set the thirteenth layer to a Convolution2D layer with 3x3 kernel, valid padding and subsample set to (1,1).  The input shape from the previous operations result in (w,h)=(5,22)
14.Use a ReLU activation function after the convolution layer.
15.Set the fifteenth layer to a Convolution2D layer with 3x3 kernel, valid padding and subsample set to (1,1).  The input shape from the previous operations result in (w,h)=(3,20)
16.Use a ReLU activation function after the previous convolution.
17.Set the seventeenth layer to a Flatten layer with the input_shape set to (1, 18, 64), which flattens it to output 1152 neurons.
18.Feed the flattened output to a Dense layer width to 100 output neurons.
19.Use a ReLU activation function after the Dense layer output.
20.Set the twentieth layer to Dense layer width to 50 output neurons. 
21.Use a ReLU activation function after the above Dense layer.
22.Set the twenty-second layer to Dense layer width to 10 output neurons. 
23.Use a ReLU activation function after the previous Dense layer.
24.Set the final layer to Dense layer width to 1 output neuron.


I use `ReLU` layers to introduce nonlinearity.

## Objective, Loss function and Hyper-Parameter tuning

I used `Mean Squared Error` as a loss metric. It seemed like the right choice as the goal is to minimize the steering angle predictions.
The training and validation loss seemed to reduce rapidly in the initial 2 epochs of batch size 64 over the entire sample set, to about 0.0443 training and about 0.0202 validation.
However, the weights on the model wouldn't be able to predict angles for all the scenarios of driving, very well. I pushed to use 256 samples over 7 epochs and ended up with 0.0413 training loss and 0.0178 validation loss.
Anymore and the loss wouldn't be lowered very much. It is also important to ensure the training loss doesn't go very close to 0 as it could signify overfitting.

I used an [`Adam`] optimizer and the learning rate wasn't tuned manually. Due to the second order moments, it performs better than RMSProp which seems to be the other option that could be used as an optimizater.
However, Adam has intrinsic benefits of RMSProp and hence is a better choice for this purpose.   

## Controlling Overfitting

I used Dropout to prevent overfitting after all the max pooling layers. The Dropout was [set to 50%]

Other ways overfitting was avoided, was by reducing bias towards lower steering angles so that the model learned to make sharp turns effectively. This was done in a generator routine that would spew batches of data.

## Image Preprocessing

The Image data is [preprocessed] in the model using the following techniques:
* Resize to a smaller `WIDTH = 200`, `HEIGHT = 66` during the image loading stage using CV2
* Brightness adjustment towards darker transformation by operating in the HSV colorspace. So it required converting from RGB to HSV and back. 
* Normalization

## Steering Angle Preprocessing
The steering angles were dithered by a static offset on left/right camerae images that were close to 0, as this provided for correction on left/right images when the car is driving straight.


# Training Strategy

## Building an initial Model with data with key insgihts 
I trained the model initially using ImageDataGenerator that performs custom image transformations such as shearing, rotation, translation.
However, while this trained the network to drive straight, I couldn't get it to turn around the sharper curves/corners. Or sometime even gradual curves would mess the steering maneuvers.
I'd get a good loss of about 0.0162, but autonmous driving wouldn't be per expectations. Reason being it doesn't provide a method to dither/modify the steering angles appropriately as it does to the images.

That forced me to resort to writing my own data generator and have control over modifying the steering angles as well. I generated my own image and angle data at the turns/corners and fed that first to my model.
The number of such additional data I created was 447 (149 each containg center/left/right corresponding to driving at the sharp left turn after the bridge). I needed this data since this was the most problematic corner where
the car would continue onto the dust road possible getting confused with the texture/color of the bridge that it just crossed.
Udacity data also didn't have too many images around this corner, for the network to learn from.

The overall strategy for deriving a model architecture was to initially overfit the model on key images and then regularize it for the entire track.
This was like a data manipulation trick I had to use to get the model weights around optimal values for further training.

Even in the data with focus on turns, I used all 3 camera images via a data aggregation trick while reading it from Pandas. That automatically guarantees the model is trained with center/left/right shuffled data
via a random choice thaat picks one of them up from the interspersed data set.  

Example of images I used are as follows:

**Recovery: Extreme Left of Lane**

![right_2017_02_09_10_16_20_158]()

**Drive Straight: Center of Lane**

![center_2016_12_01_13_31_14_295]()

**Recovery: Extreme Right of Lane**

![left_2017_02_09_10_16_15_383]()

## Data augmentation for balancing the set
Perhaps this was the biggest challenge in sanitizing the data correctly. The provided Udacity data for each of center/left/right cameras was ~8K. That totals to ~24K when using the left/right for recovery.
A majority of the data had a 0 steering angle bias. So the need to augment data containing more turns was imperative so that the model performed well for all road layouts.
Some key data transformations/augmentations performed were:
1. Flipping of left/right camera images corresponding to 0 degree steering, after dithering them by 0.2 to avoid bias driving straight.
2. In the input sample/target batch generator, I add additional translation (based on a probability of 70% of low steering angle images) and correspondingly dither the steering angle (for 99 pixels translation, I dither by 0.3.
using CV2's warpAffine for this.
3. Remove bias towards selecting small angles by adding a bias term and threshold like this. Ofcourse there is a probability of reselcting a low steering angle data combination,
but we're banking on the probability of the random choice dissuading us from such a selection. 
            STATIC_BIAS = 0.1
            threshold = np.random.uniform(0.1,0.2)
            if ((abs(batch_y[i]) + STATIC_BIAS) < threshold):
                choice = np.random.choice(len(X_train))
                batch_X[i] = brightness(X_train[choice])            
                batch_X[i] = normalize_grayscale(batch_X[i])
                batch_y[i] = y_train[choice]
4. Brightness adjust between 0.5 and 1.0 with uniform random choosing, to adjust for a darker transformation.
5. Normalize images to -0.5 to 0.5 range.
6. Nvidia arch with input images resized to 66x200 and using ReLu activation and dropout.


**Horizontally Flipped Data Histogram: Symmetric But Unbalanced**

![Histogram: Symmetric But Unbalanced](https://cloud.githubusercontent.com/assets/2206789/22631698/e0e85e38-ebc6-11e6-94cb-05f7739f188d.png)

# Acknowledgements & References
* **David Browne**, **Manav Kataria**, **Sameh Mohamed** and **Pierluigi Ferrari** - for constant discussions on the slack channel probing my ideas with questions, logic and providing useful directions.
* **Paul Hearty** - for invaluable [project tips](https://carnd-forums.udacity.com/questions/26214464/behavioral-cloning-cheatsheet) provided on Udacity forums that saved a lot of time
