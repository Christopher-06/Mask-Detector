import torch as T
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

# HYPER_PARAMETERS
training_batch_files_size = 200
training_save_intervall = 50
batch_size = 20
learning_rate = 0.0001


neural_network_file = 'data/detector_network_v2.pt'

cam_url = 'http://192.168.178.38:8080/video'

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')