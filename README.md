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
      * [Architecture: NVIDIA End-to-End Deep Learning Network](#architecture-nvidia-end-to-end-deep-learning-network)
      * [Optimizer and loss metric](#optimizer-and-loss-metric)
      * [Regularization](#regularization)
      * [Image Preprocessing](#image-preprocessing)
      * [Steering Angle Preprocessing](#steering-angle-preprocessing)
   * [Training Strategy](#training-strategy)
      * [Building an initial Model with data with key insights ](#building-an-overfitted-model-with-key-insights)
      * [Data augmentation for balancing the set](#data-augmentation-for-balancing-the-set)
   * [Acknowledgements &amp; References](#acknowledgements--references)

---

# Behavioral Cloning
Using Keras to make building deep neural networks for predicting autonomous steering angles.

# Video of the car driving around the track
This video shows the car driving around the track for the entire lap. It is recorded in "fastest" graphics setting on the simulator since my laptop that has low graphics capabilities.

[![youtube_video](https://cloud.githubusercontent.com/assets/16203244/23098803/86778cd0-f60c-11e6-8290-cd4490119b86.png)](https://www.youtube.com/watch?v=wj1BPVsaHkI)

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
* `model.ipynb` - A python notebook of model.py that can be run on jupyter.
* `README.md` - description of the development process (this file)
* video_48fps.mp4 - a recording of the car driving around track 1 in autonmous mode at 48 fps
* video_60fps.mp4 - a recording of the car driving around track 1 in autonmous mode at 60 fps
* Udacity Dataset - Track1 Dataset Used for Training. [Download here](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)
* Unity3D Simulator - [Github Repo](https://github.com/udacity/self-driving-car-sim).

Repository includes all required files and can be used to run the simulator in autonomous mode.

# Code Quality
## Functional Code
Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing
```
$ python drive.py model.h5
```
Since my system is windows with low graphics capabilities, I've recorded and tested in graphics mode "fastest" on the simulator.
In other graphics setting, my simulator doesn't behave as well since there are inherent delays in the relaying of steering commands and response. Please make sure to test the autonomous mode in "fastest" graphics setting.

## Comments inline with code

The `model.py` file contains the entire code with appropriate comments for training and saving the CNN design. It contains comments to explain how the code works.

# Model Architecture & Solution Design

## Architecture: NVIDIA End-to-End Deep Learning Network

The model consists of a convolution neural network based on architecture by ![NVIDIA](https://arxiv.org/pdf/1604.07316v1.pdf)
Namely,

* 5x5 filters. Three of them with depths of 24, 36, 48.
* Followed by two 3x3 filtes of depth 64 and 64.
* And then finally flattening the output and reducing the dimensionality via fully-connected dense layers to a steering angle prediction.

Details of padding and sampling for each convolution layer is specified with detailed comments in code. They consist of:

1. Set the first layer to a Convolution2D layer with 5x5 kernel, input_shape set to (66, 200, 3) and subsample to (1,1)
2. Use a MaxPooling2D layer that subsamples by (2,2) after the previous convolution.
3. Use a Dropout layer at 0.5 dropout, following the pooling layer.
4. Use a ReLU activation function after the Dropout layer.
5. Set the fifth layer to a Convolution2D layer with 5x5 kernel, valid padding and subsample set to (2,1). The input shape from the previous operations result in (w,h)=(31,98)
6. Use a MaxPooling2D layer that subsamples by (1,2) after the previous convolution in the fifth layer.
7. Use a Dropout layer at 0.5 dropout, following the pooling layer.
8. Use a ReLU activation function after the Dropout layer.
9. Set the ninth layer to a Convolution2D layer with 5x5 kernel, valid padding and subsample set to (1,2). The input shape from the previous operations result in (w,h)=(14,47)
10. Use a MaxPooling2D layer that subsamples by (2,1) after the previous convolution.
11. Use a Dropout layer at 0.5 dropout, following the pooling layer.
12. Use a ReLU activation function after the Dropout layer.
13. Set the thirteenth layer to a Convolution2D layer with 3x3 kernel, valid padding and subsample set to (1,1).  The input shape from the previous operations result in (w,h)=(5,22)
14. Use a ReLU activation function after the convolution layer.
15. Set the fifteenth layer to a Convolution2D layer with 3x3 kernel, valid padding and subsample set to (1,1).  The input shape from the previous operations result in (w,h)=(3,20)
16. Use a ReLU activation function after the previous convolution.
17. Set the seventeenth layer to a Flatten layer with the input_shape set to (1, 18, 64), which flattens it to output 1152 neurons.
18. Feed the flattened output to a Dense layer width to 100 output neurons.
19. Use a ReLU activation function after the Dense layer output.
20. Set the twentieth layer to Dense layer width to 50 output neurons. 
21. Use a ReLU activation function after the above Dense layer.
22. Set the twenty-second layer to Dense layer width to 10 output neurons. 
23. Use a ReLU activation function after the previous Dense layer.
24. Set the final layer to Dense layer width to 1 output neuron.

I use `ReLU` layers to introduce nonlinearity.

## Optimizer and loss metric

* I used `Mean Squared Error` as a loss metric. It seemed like the right choice as the goal is to minimize the steering angle predictions.
* The training and validation loss seemed to reduce rapidly in the initial 2 epochs of batch size 64 over the entire sample set, to about 0.0443 training and about 0.0202 validation.
    However, the weights on the model wouldn't be able to predict angles for all the scenarios of driving, very well. I ended up using 256 samples over 7 epochs and ended up with 0.0413 training loss and 0.0178 validation loss.

![network_training_summary](https://cloud.githubusercontent.com/assets/16203244/23098491/1bff0d14-f603-11e6-8149-7351531a838c.png)

* Anymore and the loss wouldn't be lowered very much. It is also important to ensure the training loss doesn't go very close to 0 as it could signify overfitting.
* `Adam` optimizer was used and thus a learning rate wasn't tuned manually. The learning rate is modified inversely proportional to the square of the gradients accumulated via an exponentially weighted moving average. 
* It is seen as a variant on the combination of RMSProp and momentum.
* Since it includes bias corrections for the first and second order moments to account for their initializations at the origin, it is a popular choice. 
* Although RMSProp incorporates the second-order momemtn estimate, it lacks the correction factor. 
  This causes it to have high bias early in training.

## Regularization

* I used Dropout to prevent overfitting after all the max pooling layers. The Dropout was set to 50%.
* Reducing bias towards lower steering angles so that the model learned to make sharp turns effectively.
  This was done in a generator routine that would spew batches of data.

## Image Preprocessing

The Image data is preprocessed in the model using the following techniques:

* Resize to a smaller `WIDTH = 200`, `HEIGHT = 66` during the image loading stage using CV2
* Brightness adjustment towards darker transformation (0.5 and 1.0 with uniform random choosing) by operating in the HSV colorspace. So it required converting from RGB to HSV and back.
  
         # Generate random brightness function, produce darker transformation 
        def brightness(image):
            #Convert 2 HSV colorspace from RGB colorspace
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            #Generate new random brightness
            rand = random.uniform(0.5,1.0)
            hsv[:,:,2] = rand * hsv[:,:,2]
            #Convert back to RGB colorspace
            new_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return new_image

* Normalize images to -0.5 to 0.5 range.

        # Normalize the data features using Min-Max scaling
        def normalize_min_max(image):
            a = -0.5
            b = 0.5
            image_min = image.min()
            image_max = image.max()
        return a + (((image - image_min) * (b - a))/ (image_max - image_min))

## Steering Angle Preprocessing
The steering angles were dithered by a static offset on left/right camerae images that were close to 0, as this provided for correction on left/right images when the car is driving straight.

# Training Strategy

## Building an initial Model with data with key insights 
* I trained the model initially using ImageDataGenerator that performs custom image transformations such as shearing, rotation, translation.
  However, while this trained the network to drive straight, I couldn't get it to turn around the sharper curves/corners.
  Or sometime even gradual curves would mess the steering maneuvers.
* I'd get a good loss of about 0.0162, but autonmous driving wouldn't be per expectations.
  Reason being Keras' ImageDataGenerator doesn't provide a method to dither/modify the target outputs (steering angles) appropriately as it does to the images.
* That forced me to resort to writing my own data generator and have control over modifying the steering angles as well.
  I generated my own image and angle data at the turns/corners and fed that first to my model.
* The number of such additional data I created was 447 (149 each containg center/left/right corresponding to driving at the sharp left turn after the bridge).
* I needed this data since the most problematic corner was where the car would continue onto the dust road without making a sharp left,  possibly getting confused with the texture/color of the bridge that it just crossed.
  Udacity data also didn't have too many images around this corner, for the network to learn from.
* The overall strategy for deriving a model architecture was to initially overfit the model with images on key turns on the track and then regularize it for the entire track.
* I used all 3 camera images via a data aggregation trick while reading it from Pandas.
  That automatically guarantees the model is trained with center/left/right shuffled data via a random choice that picks one of them up from the interspersed data set.  

Example of images I used are as follows:

**Sample Recovery from extreme left of lane**

* The sharp turns needed more training data to train the network around these corners. Here's a recovery from extreme left of the lane.

![right_2017_02_09_10_16_20_158](https://cloud.githubusercontent.com/assets/16203244/23098492/1bffa4fe-f603-11e6-8732-d7b692fc0e2a.jpg)

**Drive Straight: Center of Lane**

![center_2016_12_01_13_31_14_295](https://cloud.githubusercontent.com/assets/16203244/23098488/1bfdc616-f603-11e6-9136-e520fd359c96.jpg)

**Sample Recovery from extreme right of lane**

* Here's a similar recovery from extreme right of the lane on the same turn. The network trainied with such data helped converge well around sharp turns.

![left_2017_02_09_10_16_15_383](https://cloud.githubusercontent.com/assets/16203244/23098490/1bfe2a16-f603-11e6-8a53-7c1c46bdd8f5.jpg)

## Data augmentation for balancing the set
* Perhaps the biggest challenge in sanitizing the data correctly. 
  The Udacity data for each of center/left/right cameras was ~8K. That totals to ~24K when using the left/right for recovery.
* A majority of the data had a 0 steering angle bias. So the need to augment data containing more turns was imperative so that the model performed well for all road layouts.

Some key data transformations/augmentations performed were:

1. Adding a steering dither for correction on images to avoid bias in driving straight.

        # Input pre-processing step - dither the left and right steering angles by a small amount for recovery
        STEERING_DITHER = 0.2

        ## Additional training data at the turns - Left image's steering angle dither by 0.2
        y_train_turns[1::3] = [x + STEERING_DITHER for x in y_train_turns[1::3]]

        ## Additional training data at the turns - Right image's steering angle dither by 0.2
        y_train_turns[2::3] = [x - STEERING_DITHER for x in y_train_turns[2::3]]
        
        ## Original Udacity training data - Dither the right camera image steering angle by 0.2, for 0 steering angle inputs
        y_train[2::3] = [x - STEERING_DITHER * (x == 0) for x in y_train[2::3]]

        ## Original Udacity training data - Dither the left camera image steering angle by 0.2, for 0 steering angle inputs
        y_train[1::3] = [x + STEERING_DITHER * (x == 0) for x in y_train[1::3]]

2. Flipping of left/right camera images and angles to produce more balanced data, on low angle of steering samples.

        X_train_turns_flipped[0::2] = [np.fliplr(x) for x in X_train_turns[1::3]]
        X_train_turns_flipped[1::2] = [np.fliplr(x) for x in X_train_turns[2::3]]
        y_train_turns_flipped[0::2] = [x* (-1) for x in y_train_turns[1::3]]
        y_train_turns_flipped[1::2] = [x* (-1) for x in y_train_turns[2::3]]

        X_train_flipped[0::2] = [np.fliplr(x) for x in X_train[1::3]]
        X_train_flipped[1::2] = [np.fliplr(x) for x in X_train[2::3]]
        y_train_flipped[0::2] = [x* (-1) for x in y_train[1::3]]
        y_train_flipped[1::2] = [x* (-1) for x in y_train[2::3]]

3. In the input sample/target batch generator, I add additional translation (based on a probability of 70% of low steering angle images) and correspondingly dither the steering angle (for 99 pixels translation, I dither by 0.3 using CV2's warpAffine for this).

            # Apply horizontal translation to low steering angles (< 0.1) to 70% of qualified images
            translate_prob = np.random.uniform(0,1)
            if (abs(batch_y[i]) < 0.1 and translate_prob >= 0.3):
                angle = np.random.uniform(0,1)
                pixels = int((TRANSLATION_X_RANGE * angle)/TRANSLATION_Y_RANGE)
                if (batch_y[i] > 0):
                    batch_X[i] = horizontal_translation(batch_X[i], pixels)
                    batch_y[i] = batch_y[i] - angle
                else:
                    batch_X[i] = horizontal_translation(batch_X[i], -pixels)
                    batch_y[i] = batch_y[i] + angle

4. Remove bias towards selecting small angles by adding a bias term and threshold like this.
  Ofcourse there is a probability of reselcting a low steering angle data combination.
  However, we're banking on the probability of the random choice dissuading us from such a selection. 
  
            STATIC_BIAS = 0.1
            threshold = np.random.uniform(0.1,0.2)
            if ((abs(batch_y[i]) + STATIC_BIAS) < threshold):
                choice = np.random.choice(len(X_train))
                batch_X[i] = brightness(X_train[choice])            
                batch_X[i] = normalize_grayscale(batch_X[i])
                batch_y[i] = y_train[choice]

**Flipped Data along vertical axis Histogram: Symmetric But Unbalanced to produce more data with non-trivial steering angles**

![histogram_of_data_with_flips](https://cloud.githubusercontent.com/assets/16203244/23098489/1bfe1d78-f603-11e6-815f-e595759356e5.png)

# Acknowledgements & References
* **David Browne**, **Manav Kataria**, **Sameh Mohamed** and **Pierluigi Ferrari** - for constant discussions on the slack channel probing my ideas with questions, logic and providing useful directions.
* **Paul Hearty** - for invaluable [project tips](https://carnd-forums.udacity.com/questions/26214464/behavioral-cloning-cheatsheet) provided on Udacity forums that saved a lot of time
