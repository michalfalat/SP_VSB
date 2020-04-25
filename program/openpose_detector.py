import sys
import cv2
import os
from sys import platform

opWrapper = None
datum = None


def initialize(model = "MPI"):
    try:
        # Import Openpose (Windows/Ubuntu/OSX)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
            # Windows Import
            if platform == "win32":
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append('C:/Diploma/openpose/build/python/openpose/Release')
                os.environ['PATH'] = os.environ['PATH'] + ';C:/Diploma/openpose/build/x64/Release;C:/Diploma/openpose/build/bin;'
                import pyopenpose as op
            else:
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append('../../python')
                # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
                # sys.path.append('/usr/local/python')
                from openpose import pyopenpose as op
        except ImportError as e:
            print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e


        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = "C:/Diploma/openpose/models/"
        params["model_pose"] = model

        # Starting OpenPose
        print("Configuring openpose..")
        global opWrapper
        global datum
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        print("Configuration done.")
        # Process Image
        datum = op.Datum()

        print("Dataset " + model +  " - List of body parts:")
        poseModel = None
        if(model == "MPI"):
            poseModel = op.PoseModel.MPI_15
        elif (model == "COCO"):
            poseModel = op.PoseModel.COCO_18
        print(op.getPoseBodyPartMapping(poseModel))
    except Exception as e:
        print(e)
        sys.exit(-1)


def detect_op_pose(frame, params):
    global datum
    global opWrapper
    if opWrapper is None:
        initialize("COCO")
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])

    if params.showNativeOutput is True:
        cv2.imshow("OP_POSE native output result", datum.cvOutputData[:, :, :])
    return datum.poseKeypoints
