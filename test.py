import cv2
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt


tf_path = './data_alter/gonogo/Realigned MRI video.1.tiff'
video_tf = tf.imread(tf_path)

cap = cv2.VideoCapture('./data_alter/gonogo/cv2_video1.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./data_alter/gonogo/cv2_video1.mp4', fourcc, 20.0, (256, 256), False)
for frame in video_tf:
    frame = frame.astype(np.uint8)
    out.write(frame)
out.release()
cv2.destroyAllWindows()
