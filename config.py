import torch as T
from torchvision import transforms
import numpy as np
import cv2


transform = transforms.Compose([
    transforms.ToTensor()
])

# HYPER_PARAMETERS
training_batch_files_size = 200
training_save_intervall = 50
batch_size = 20
learning_rate = 0.0001

show_webcam_images = True
blank_image = np.ones((512,512,3))
no_video_stream = cv2.putText(blank_image, 'No video stream', (200+0*int(512/2), int(512/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)

neural_network_file = 'data/detector_network_v2.pt'

webcams_filename = 'webcams.json'

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')