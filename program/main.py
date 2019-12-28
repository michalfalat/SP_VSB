"""Main file"""
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from video_helper import process_video


def main():
    """main"""
    process_video('car_videos\\output\\fusion2.mp4', False, 900, 59, "qqq")
    # process_video('videos\\auto_7.mp4', False)


main()
