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
* TF Pose estimation installation  + datasets to use (more info on [official TF Pose estimation GitHub](https://github.com/ildoonet/tf-pose-estimation))
* Lot of luck :) 


# Driver Analysator
configuration is built on console arguments. Basic command for starting program is:


## Basic usage
``` 
python main.py  --video-input testVideo.mp4
```

## List of existing arguments for main.py
```
--video-input  <value>              // input video name (default: testVideo.mp4)
--framework  <value>                // framework to use (TF_POSE or OP_POSE) (default: OP_POSE)
--model_name  <value>               // trained model name from neural network (default: defaultModel.data)
--save-train_image                  // save frames for neural network training
--save-train_image_path  <value>    // path for saving train image
--op-dataset  <value>               // MPI or COCO (default: COCO)
--init-frame  <value>               // starting frame of video (default: 0)
--record-video                      // if proceeded video should be saved to filesystem 
--record-video-name  <value>        // name for saved video (default 'output.mp4')
--record-video-fps  <value>         // what FPS should be used for saved video (default: 30)
--show-native-output                // if default format from framework should be shown
--resolution-height   <value>       // resolution height for videos (default: 800)
--no-final-statistics               // disable final stats
--no-continuos-statistics           // disable continous stats
--no-image-print-statistics         // disable printing  stats directly to the image
--no-head-orientation-detection     // disable head orientation detection
--no-head-orientation-limit         // disable head orientation helper lines for allowed range
--no-seat-belt-detection            // not used 
--no-body-detection                 // disable pose detection
--no-showing-output                 // disable showing images during processing
--no-nn-filtering                   // if results of neural network should not be filtered (worse accuracy)
```



# Training neural network 
configuration is built on console arguments. Basic command for starting program is:

## Basic usage
``` 
python train_nn.py
```

## List of existing arguments for train_nn.py
```
--model_name  <value>               // trained model name from neural network (default: defaultModel.data)
--path  <value>                     // path to folder, which contains training images (default: './program/train_nn/')
--epochs  <value>                   // number of training epochs (default: 12)
```

## More info will be added soon...

Created in 2019-2020. Everything is free to use.