import torch as T
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image
import cv2
from threading import Thread
import time
import os
from os import path, listdir
import config as conf
from tqdm import tqdm
import helper

if T.cuda.is_available():
    T.cuda.empty_cache()


def save(model):
    model.cpu()
    # save only weights & co
    T.save(model.state_dict(), conf.neural_network_file)
    model.to(conf.device)

def load():
    model = MaskDetector()
    if path.isfile(conf.neural_network_file):
        # load file if it is available
        # only weights and stuff like that
        model.load_state_dict(T.load(conf.neural_network_file, map_location=conf.device))
        print('Loaded data: ' + conf.neural_network_file)
    
    return model.to(conf.device)

def has_mask(input_tensor):
    # convert tensor to a simple boolean
    if T.argmax(input_tensor).item() == 1:
       return True
    return False

class MaskDetector(nn.Module):
    def __init__(self):
        super(MaskDetector, self).__init__()        

        # conv Layers
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, stride=1)
        self.conv2_bn = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=3, stride=1)
        self.conv3_bn = nn.BatchNorm2d(30)
        self.conv4 = nn.Conv2d(30, 40, kernel_size=3, stride=1)
        self.conv4_bn = nn.BatchNorm2d(40)

        # Drop out
        # --> Better training performance
        self.drop1 = nn.Dropout2d(p=0.4)

        # fully Connected Layers
        self.fc1 = nn.Linear(480,250)
        self.fc2 = nn.Linear(250, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        # conv
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_bn(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.conv4_bn(self.conv4(x)), 2))

        # flat it
        x = self.drop1(x)
        x = x.view(-1, 480)

        # Fully connected
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # [0, 1] --> Mask
        # [1, 0] --> No Mask
        return F.softmax(x)

class Trainer():
    def __init__(self, model : MaskDetector, epochs : int):
        self.model = model
        self.epochs = epochs

        self.optimizer = optim.Adam(self.model.parameters(), lr=conf.learning_rate)
        self.criterion = nn.BCELoss() 

        self.files = helper.get_file_lists()
        self.batches = []
        self.batch_index = 0

    def get_batch(self):
        files = listdir('data/train/tensors/')
        if len(files) == 0 or len(files) == 1 and '.md' in files[0]:
            # be sure to have some batch files
            print('No tensor files avaible. Please make some...')
            return None        

        # choose one random file
        file_name = np.random.choice(listdir('data/train/tensors/'))
        while '.tensor' in file_name == False:
            # be sure to have a tensor file
            file_name = np.random.choice(listdir('data/train/tensors/'))
        tensor_file = T.load('data/train/tensors/' + file_name)

        tensors, targets = [], []
        while len(tensors) < conf.batch_size:
            # get random index and insert them
            # into the lists
            index = np.random.randint(0, len(tensor_file[0]))
            tensors.append(tensor_file[0][index])
            targets.append(tensor_file[1][index])

        # return tuple
        return (T.stack(tensors), T.stack(targets))

    def run(self):      
        self.model.train()  
        for e in range(self.epochs):
            try:              
                total_batch_count = helper.get_batch_count()
                for i in range(helper.total_batch_count):
                    start = time.time()
                    data = self.get_batch()

                    if data is None:
                        break

                    # seperate data and calculate values
                    _input = data[0].to(conf.device)
                    _output = self.model(_input)
                    _target = data[1].to(conf.device)

                    # optimize network
                    self.optimizer.zero_grad()
                    loss = self.criterion(_output, _target)
                    loss.backward()
                    self.optimizer.step()

                    # status statement
                    end = time.time()
                    print(f'Epoch: {e+1}/{self.epochs}   Batch Index: {i}/{total_batch_count}   Loss: {loss}   Time: {end - start}') 
                    
                    if i % conf.training_save_intervall:
                        # save every ...
                        save(self.model)
                                            
            except KeyboardInterrupt:
                # User pressed Crlt-C
                print("User Keyboard Interrupt: Saving")
                break

        save(self.model)
        return self.model

class Tester:
    def __init__(self, model):
        self.model = model

    def run(self):
        self.model.eval()
        labels = listdir('data/test/')
        correct, wrong = 0, 0

        for l in labels:
            files = listdir('data/test/' + l)
            for f in files:
                for face in helper.detect_faces(cv2.imread('data/test/' + l + '/' + f))[0]:    
                    # get predicted values, 
                    # if the face wears a mask                
                    _output = self.model(conf.transform(face).to(conf.device).unsqueeze(0))

                    predict = 'No'
                    if 'with_mask' in l:
                        if has_mask(_output):
                            correct += 1
                            predict = 'Yes'
                        else:
                            wrong += 1

                    if 'no_mask' in l:
                        if has_mask(_output) == False:
                            correct += 1
                        else:
                            wrong += 1
                            predict = 'Yes'
        
                    # show result
                    print(f'Label: {l}   Tensor: {_output}   File: {f}   Prediction: {predict}')
                    cv2.imshow(predict, face)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    
        # end result
        print(f'Images Tested: {wrong + correct}   Correct: {correct}   Wrong: {wrong}   Percentage: {int(correct / float(correct + wrong) * 100.)}%')