import torch as T
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

training_batch_files_size = 150
training_save_intervall = 50
batch_size = 20
learning_rate = 0.0000001
neural_network_file = 'data/detector_network_v1.pt'

cam_url = 'http://192.168.178.38:8080/video'

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')