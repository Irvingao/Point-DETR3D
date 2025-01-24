import cv2
from cv2 import VideoWriter_fourcc
import os

img_dir = "/home/niyiyang/code/detr3d/work_dirs/bev"
savepath = "/home/niyiyang/code/detr3d/work_dirs/bev.mp4"
fourcc = VideoWriter_fourcc(*"mp4v")
img_list = os.listdir(img_dir)

img_size = cv2.imread(os.path.join(img_dir, img_list[0])).shape[:2]
print(img_size)
VW = cv2.VideoWriter(savepath, fourcc, 10, img_size)
for img in img_list:
    VW.write(cv2.imread(os.path.join(img_dir, img)))
VW.release()