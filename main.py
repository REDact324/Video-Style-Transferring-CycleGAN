import torchvision.transforms as transforms
import argparse
import os
import cv2
from PIL import Image
import torch

from src.model import Generator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_args_parser():
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument('--video_path', help='directory of output text file')
  parser.add_argument('--output_path', help='input_ads the IP address or a directory of text file')

  return parser

net_GAtoB = Generator().to(device)
net_GBtoA = Generator().to(device)

size = 256

net_GAtoB.load_state_dict(torch.load('./models/net_GAtoB.pth'))
net_GBtoA.load_state_dict(torch.load('./models/net_GBtoA.pth'))

net_GAtoB.eval()
net_GBtoA.eval()

input_A = torch.ones(1, 3, 256, 256, dtype=torch.float).to(device)
input_B = torch.ones(1, 3, 256, 256, dtype=torch.float).to(device)

transforms_ = [
               transforms.Resize(int(size * 1.12), Image.BICUBIC),
               transforms.RandomCrop(size),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
               ]

video_path = '/content/IMG_5342.JPG'

def extract_frames(video_path, output_path):
  video_capture = cv2.VideoCapture(video_path)

  if not video_capture.isOpened():
    print(f"Error opening video file {video_path}")
    return

  # Get the total number of frames in the video
  total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
  print(f"Total frames in video: {total_frames}")

  fps = video_capture.get(cv2.CAP_PROP_FPS)
  frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
  frame_size = (frame_width, frame_height)

  # Create the output folder if it doesn't exist
  if not os.path.exists(output_path):
      os.makedirs(output_path)

  # Define the codec and create VideoWriter object
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

  while True:
    # Read a frame from the video
    ret, frame = video_capture.read()

    if not ret:
      break  # End of video

    frame = frame.convert('RGB')
    frame = transforms.Compose(transforms_)(frame)

    real = torch.tensor(input_A.copy_(frame), dtype=torch.float).to(device)
    fake_frame = 0.5 * (net_GBtoA(real).data + 1.0)

    out.write(fake_frame)

  video_capture.release()
  out.release()
  print(f"Processed video saved to {output_path}")

if __name__ == '__main__':
  parser = argparse.ArgumentParser('Convert video to anime style.', parents=[get_args_parser()])
  args = parser.parse_args()

  video_path = args.video_path
  output_path = args.output_path

  extract_frames(video_path, output_path)
