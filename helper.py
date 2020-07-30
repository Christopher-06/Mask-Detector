from network import *
import config as conf

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
from tqdm import tqdm

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


# *** AGENTS ***

class WebcamAgent():
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

    def do_test(self):
        length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        p_bar = tqdm(total=length, desc='Test ' + self.filename)
        self.model.eval()

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

        self.cap.release()
        if self.out is not None:
            self.out.release()

    def do_train(self, has_mask : bool, save_model = True, show_p_bar = True):
        # Prepare progress bar
        length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if show_p_bar:
            p_bar = tqdm(total=length, desc='Train ' + self.filename)

        # Define target
        _target = T.tensor([0, 1] if has_mask else [1, 0], dtype=T.float32).to(conf.device)

        # Training variables
        optimizer = optim.Adam(self.model.parameters(), lr=conf.learning_rate)
        criterion = nn.BCELoss()
        self.model.train()

        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            
            if ret == False:
                if show_p_bar:
                    p_bar.close()
                break

            # Detect all Faces and iterate through
            (faces, boxes) = detect_faces(frame)   
            for index, box in enumerate(boxes):
                face = faces[index]

                if face is not None:
                    # Get start and end coordinates
                    startX, startY, endX, endY = box[0], box[1], box[2], box[3]

                    # Calculate Value
                    _input = conf.transform(face).to(conf.device).unsqueeze(0)
                    _output = self.model(_input)    
                    
                    # Optimize network
                    optimizer.zero_grad()
                    loss = criterion(_output, _target)
                    loss.backward()
                    optimizer.step()

            if show_p_bar:
                p_bar.update(1)

        if save_model:
            save(self.model)

        self.cap.release()
        return self.model

