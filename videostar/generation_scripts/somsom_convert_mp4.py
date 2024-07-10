import math
import os
import argparse
import cv2
from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser('conversion from webm to mp4', add_help=False)
    parser.add_argument('--chunk_inference', default=False, action='store_true')
    parser.add_argument('--num_chunks', default=1, type=int, help='Number of videos per chunk')
    parser.add_argument('--chunk_index', default=0, type=int, help='Index of the chunk to process')
    return parser


def convert_webm_to_mp4(webm_path, mp4_path):
    # Capture the input video
    cap = cv2.VideoCapture(webm_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec used for mp4 format
    out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))

    # Read frames from the webm file and write them to the mp4 file
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    # Release resources
    cap.release()
    out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Video-LLaVA reasoning generate', parents=[get_args_parser()])
    args = parser.parse_args()
    input_dir = "data/something-something-v2/videos"
    output_dir = "data/something-something-v2/star_videos_new"
    videos = os.listdir(input_dir)
    
    total_videos = len(videos)
    chunk_size = total_videos//args.num_chunks
    start_index = args.chunk_index * chunk_size
    end_index = min(start_index + chunk_size, total_videos)
    selected_videos = videos[start_index:end_index]
    print(start_index, end_index)
    
    for v in tqdm(selected_videos):
        convert_webm_to_mp4(os.path.join(input_dir, v), os.path.join(output_dir, v.replace(".webm",".mp4")))        