# SP_VSB - Diploma Thesis
This repository contains set of tools to analyze  driver's behaviour during drive.
## Parts of program
* detecting pose with OpenPose or TF Pose Estimation
* Analysing pose with convolutional neural network in Keras framework
* detecting head angle  of driver based on 2D to 3D transformation

## Example of detection + evaluation of pose
![nnresult2](https://raw.githubusercontent.com/michalfalat/SP_VSB/master/images/nnresult2.png)

## Example of head angle orientation with limits
![headLimits](https://raw.githubusercontent.com/michalfalat/SP_VSB/master/images/headLimits.png)


## Required tools
* Python 3.3 or Higher
* OpenCV
* Tensorflow 1.14 (version 2.0  is not supported)
* OpenPose instalation + datasets to use (more info on [official TF Pose estimation GitHub](https://github.com/CMU-Perceptual-Computing-Lab/openpose))
* TF Pose estimation instalation  + datasets to use (more info on [official TF Pose estimation GitHub](https://github.com/ildoonet/tf-pose-estimation))