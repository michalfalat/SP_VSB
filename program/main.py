"""Main file"""
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from video_helper import process_video


def main():
    """main"""
    process_video('car_videos\\output\\15.mp4', True)
    # process_video('videos\\auto_7.mp4', False)


main()
