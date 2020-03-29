from nn import run_train, run_test
from video_helper import process_video
import argparse
"""Main file"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    """main"""
    parser = argparse.ArgumentParser(description='Program for analysing driver\'s behaviour in vehicle')
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    #required.add_argument('--type', dest='programType', action='store',  help='Determine which action should be  proceeded (TRAIN_NN, TEST_NN, PROCESS_VIDEO)', required=True)
    optional.add_argument('--framework', dest='framework', action='store', help='Determine, which framework should be used  (TF_POSE or OP_POSE) (default: TF_POSE)', default='OP_POSE')
    optional.add_argument('--video-input', dest='videoInput', action='store', help='Absolute path for input video (default: testVideo.mp4)', default='testVideo.mp4')
    optional.add_argument('--init-frame', dest='initFrame', action='store', type=int, help='Determine starting frame of video (default: 0)', default=0)
    optional.add_argument('--record-video', dest='recordVideo', action='store_true', help='Determine, if video should be recorded and saved. Use with --video-output (default: False)', default=False)
    optional.add_argument('--show-native-output', dest='showNativeOutput', action='store_true', help='Determine, if picture should be shown  in native format from framework result (default: False)', default=False)
    optional.add_argument('--video-output', dest='videoOutput', action='store', help='Absolute path for output video (default: output.mp4)', default='output.mp4')
    optional.add_argument('--resolution-height', dest='resolutionHeight', action='store', type=int, help='Resolution height for processing video in px (default: 800)', default=800)
    optional.add_argument('--fps', dest='videoFps',  action='store', help='Frames per seconds for of video (default: 30)', default=30)
    optional.add_argument('--no-final-statistics', dest='printFinalStatistics', action='store_false', help='Determine, if any statistics should be printed (default: False)', default=True)
    optional.add_argument('--no-continuos-statistics', dest='printContinuosStatistics', action='store_false', help='Determine, if any statistics should be printed (default: False)', default=True)
    optional.add_argument('--no-image-print-statistics', dest='imagePrintStatistics', action='store_false', help='Determine if head orientation should not be detected', default=True)
    optional.add_argument('--no-head-orientation-detection', dest='detectHeadOrientation', action='store_false', help='Determine if head orientation should not be detected', default=True)
    optional.add_argument('--no-seat-belt-detection', dest='detectSeatBelt', action='store_false', help='Determine if program should not detect seatbelt status', default=True)
    optional.add_argument('--no-showing-output', dest='showOutput', action='store_false', help='Determine if result images should not be shown during processing', default=True)
    optional.add_argument('--no-nn-filtering', dest='useFiltering', action='store_false', help='Determine if neural network result should not be filtered', default=True)
    args = parser.parse_args()
    # run_train()
    process_video(args)
    # run_test()
    # process_video('videos\\auto_7.mp4', False)


main()
