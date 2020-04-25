# Diploma Thesis - Driver Analysis Using Spherical Cameras
This repository contains set of tools to analyze  driver's behaviour during drive. Pose was captured with spherical camera GoPro Fusion
## Parts of the program
* detecting pose with OpenPose or TF Pose Estimation
* unifying results of both frameworks for next usage
* analysing pose with convolutional neural network in Keras framework
* detecting head angle of driver based on 2D to 3D transformation

## Example of detection + evaluation of pose
![nnresult2](https://raw.githubusercontent.com/michalfalat/SP_VSB/master/images/nnresult2.png)

## Example of head angle detection with limits
![headLimits](https://raw.githubusercontent.com/michalfalat/SP_VSB/master/images/headLimits.png)


## Required tools
* Python 3.3 or Higher
* OpenCV
* Tensorflow 1.15 (version 2.0  is not supported)
* Keras 2.3.1
* CUDA 10.1
* OpenPose instalation + datasets to use (more info on [official Openpose GitHub](https://github.com/CMU-Perceptual-Computing-Lab/openpose))
* TF Pose estimation instalation  + datasets to use (more info on [official TF Pose estimation GitHub](https://github.com/ildoonet/tf-pose-estimation))
* Lot of luck :) 


# Usage
configuration is built on console arguments. Basic command for starting program is:

``` 
python main.py  --video-input testVideo.mp4 --record-video
```

## List of existing arguments
```
--video-input'                        // input video name
--framework'                          // framework to use (TF_POSE or OP_POSE)
--model_name'                         // trained model name from neural network 
--save-train_image'                   // if frame of video should be saved  for neural network training
--save-train_image_path'              // path for saving train image
--op-dataset'                         // MPI or COCO 
--init-frame'                         // starting frame of video (default 0)
--record-video'                       // if proceeded video should be saved to filesystem 
--record-video-name'                  // name for saved video (default 'output.mp4')
--record-video-fps'                   // what FPS should be used for saved video
--show-native-output'                 // if default format from framework should be shown
--resolution-height'                  // resolution height for videos
--no-final-statistics'                // disable final stats
--no-continuos-statistics'            // disable continous stats
--no-image-print-statistics'          // disable printing  stats directly to the image
--no-head-orientation-detection'      // disable head orientation detection
--no-head-orientation-limit'          // disable head iroentation helper lines  for  range
--no-seat-belt-detection'             // not used 
--no-body-detection'                  //
--no-showing-output'                  //
--no-nn-filtering'                    // if results of neural network should not be filtered (worse accuracy)
```

## More info will be added soon...

Created in 2019-2020. Everything is free to use.