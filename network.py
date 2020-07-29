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

def get_file_lists():
    # list all training photos
    return  [(i, [1, 0]) for i in listdir('data/train/no_mask')] + [(i, [0, 1]) for i in listdir('data/train/with_mask')] 

def get_batch_count():
    # calculate total avaible batches
    files = len(listdir('data/train/no_mask')) + len(listdir('data/train/with_mask'))
    return int(files / conf.batch_size)

def detect_faces(img):  
    if img is None:
        # Error prehandling
        return ([], [])    
        
    # load Caffe model and prepare image
    net = cv2.dnn.readNetFromCaffe('data/deploy.prototxt.txt', 'data/face_detection.caffemodel')
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()
    faces, boxes = [], []
    for i in range(0, detections.shape[2]):
        # iterate through every face
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            # face is ok
            # --> get box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")                                           
            face = img[startY:endY, startX:endX]

            if face is not None:
                try:
                    # try to resize
                    face = cv2.resize(face, (80, 100))
                    faces.append(face)
                    boxes.append((startX, startY, endX, endY))
                except:
                    # img slice gives some shit back
                    pass
                
    return (faces, boxes)

def make_training_batch_files():        
    # delete old
    for file in listdir('data/train/tensors/'):
        os.remove('data/train/tensors/' + file)

    # make them   
    data = get_file_lists()
    targets, tensors = [], []
    masked_faces, normal_faces = 0, 0
    while len(data) > 0:
        # get one random and remove it
        index = np.random.randint(0, len(data))        
        file = data[index][0]
        target = data[index][1]
        data.pop(index)

        print(f'Index:{len(tensors)}   File: {file}')
        
       # if '.db' in file:
        #    continue

        if T.cuda.is_available():
            # clear cache
            T.cuda.empty_cache()

        if '.jpg' in file or '.png' in file or '.jpeg' in file:
            # that's an image
            if target == [0, 1]:
                img = cv2.imread('data/train/with_mask/' + file)
            else:
                img = cv2.imread('data/train/no_mask/' + file)

            for face in detect_faces(img)[0]:
                targets.append(target)
                tensors.append(conf.transform(face))

                if target == [0, 1]:
                    masked_faces += 1
                else:
                    normal_faces += 1

        if len(data) == 0 or len(targets) >= conf.training_batch_files_size:
            # tensor_files max capacity reached
            # or the last one was calculated
            T.save((tensors, T.Tensor(targets)), f'data/train/tensors/T{str(np.random.randint(100,1000))}.tensor')
            targets, tensors = [], []
            print(f"--> Made one. Masked: {masked_faces}   Unmasked: {normal_faces}")

    # end statement
    print("finished")
    print(f'Masked: {masked_faces}   Unmasked: {normal_faces}')

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

        self.files = get_file_lists()
        self.batches = []
        self.batch_index = 0

    def get_batch(self):
        if len(listdir('data/train/tensors/')) == 0 or '.md' in listdir('data/train/tensors/')[0]:
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
                total_batch_count = get_batch_count()
                for i in range(total_batch_count):
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
                for face in detect_faces(cv2.imread('data/test/' + l + '/' + f))[0]:    
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

class Webcam_Agent():
    def __init__(self, model, cam_url, name = '(Unknown)', run_thread_start = False):
        self.model = model        
        self.cam_url = cam_url
        self.name = name
        self.last_img = None
        self.proceed_img = None
        self.cam_alive = False
        # Stream Thread
        self.streaming_thread = Thread(target=self.get_stream)
        self.streaming_thread.daemon = True
        self.streaming_thread.start()

        if run_thread_start:
            # Run Thread
            self.run_thread = Thread(target=self.run)
            self.run_thread.daemon = True
            self.run_thread.start()

    def get_stream(self):     
        # get current image from webcam   
        while True:
            try:       
                if self.last_img is None:
                    # if no connection is available or an error occured,
                    # the image will become None:
                    # Then reinit the capture
                    cap = cv2.VideoCapture(self.cam_url)
                    self.cam_alive = False

                _, self.last_img = cap.read()                                
            except:
                # restart connection
                cap = cv2.VideoCapture(self.cam_url)
                self.cam_alive = False

    def run(self):     
        self.model.eval()   
        while True:
            while self.last_img is None:
                # wait for an image 
                # None, if stream not started yet or
                # problems occured
                time.sleep(0.1)

            # get a copy of the image(-array)
            # detect all faces
            img = self.last_img.copy()
            self.cam_alive = True
            (faces, boxes) = detect_faces(img)   


            for index, box in enumerate(boxes):
                # perform on every face
                face = faces[index]

                if face is not None:
                    # get start and end coordinates. As well the point where the text should stand
                    startX, startY, endX, endY = box[0], box[1], box[2], box[3]
                    y = startY - 10 if startY - 10 > 10 else startY + 10

                    # test if a mask is anywhere
                    _input = conf.transform(face).to(conf.device).unsqueeze(0)
                    _output = self.model(_input)    
                    if has_mask(_output):
                        # masked person --> Blue border
                        img = cv2.rectangle(img, (startX, startY), (endX, endY), (255, 0, 0), thickness=3)
                        cv2.putText(img, 'Masked', (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
                    else:
                        # unmasked person --> Red border
                        img = cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), thickness=3)
                        cv2.putText(img, 'No mask', (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                        
            self.proceed_img = cv2.resize(img, (512, 512))

class VideoAgent():
    def __init__(self, model : MaskDetector, filename):
        self.filename = filename
        self.cap = cv2.VideoCapture('data/test/video/' + filename)
        self.model = model
        self.out = None

    def do_job(self):
        length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(length)
        p_bar = tqdm(total=length, desc=self.filename)
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            
            if ret == False:
                p_bar.close()
                print('Video ending. Saving for ' + self.filename)
                break

            if self.out is None:
                # Create video writer
                height, width, _ = frame.shape
                self.out = cv2.VideoWriter('data/test/video/out_' + self.filename + '.avi', 
                            cv2.VideoWriter_fourcc(*'XVID'),
                            20.0, (width, height))   

            (faces, boxes) = detect_faces(frame)   
            for index, box in enumerate(boxes):
                # perform on every face
                face = faces[index]

                if face is not None:
                    # get start and end coordinates. As well the point where the text should stand
                    startX, startY, endX, endY = box[0], box[1], box[2], box[3]
                    y = startY - 10 if startY - 10 > 10 else startY + 10

                    # test if a mask is anywhere
                    _input = conf.transform(face).to(conf.device).unsqueeze(0)
                    _output = self.model(_input)    
                    if has_mask(_output):
                        # masked person --> Blue border
                        frame = cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), thickness=3)
                        cv2.putText(frame, 'Masked', (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
                    else:
                        # unmasked person --> Red border
                        frame = cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), thickness=3)
                        cv2.putText(frame, 'No mask', (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            self.out.write(frame)
            p_bar.update(1)
           # cv2.imshow('video', cv2.resize(frame,(int(width/3), int(height/3))))
            #cv2.waitKey()
            #cv2.destroyAllWindows()

        self.cap.release()
        if self.out is not None:
            self.out.release()