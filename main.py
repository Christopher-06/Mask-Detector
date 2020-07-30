from network import *
from helper import *
import config as conf
import argparse
import os, time
import json
import matplotlib.pyplot as plt
import cv2

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-train', help='Start a training session. Specify the epochs', type=int)
parser.add_argument('-test', help='Test all images and get result', action='store_true')
parser.add_argument('-webcam', help='Streaming a webcam', action='store_true')
parser.add_argument('-newbatches', help='make new batch files (Training porpuse)', action='store_true')
parser.add_argument('-video', help='test an entire video', action='store_true')
args = parser.parse_args()

def main():
    model = load()    

    model = VideoAgent(model, 'No_Masked.mp4').do_train(False)  
    model = VideoAgent(model, 'With_Mask.mp4').do_train(True)  


    if args.newbatches:
        make_training_batch_files()
    if args.train:
        model = Trainer(model, args.train).run()
    if args.test:
        Tester(model).run()
    if args.video:
        videos = os.listdir('data/test/video/')
        if len(videos) == 0:
            # Error if no videos are available
            print('No videos found in data/test/video/. Please add any')

        for vid in videos:
            if '.mp4' in vid or '.avi' in vid:
                if not 'out_' in vid:
                    VideoAgent(model, vid).do_test()       

    if args.webcam:
        if os.path.isfile(conf.webcams_filename) is False:
            # Error prehandling
            print(f"'{conf.webcams_filename}' does not exist. Please create one and/or adjust the config file")
            exit()

        # Loading the file
        print(f"Loading {conf.webcams_filename}")
        try:
            f = open(conf.webcams_filename)
            obj = json.loads(f.read())
            f.close()
        except:
            print("Failed to load file. Exit")
            exit()

        agents = []
        for cam in obj:
            # Start all agents
            agent = WebcamAgent(model, cam['url'], cam['name'], run_thread_start=True)
            agents.append(agent)


        while True:
            if conf.show_webcam_images:                
                for agent in agents:
                    # Prepare images
                    if agent.cam_alive and agent.proceed_img is not None:
                        cv2.imshow(agent.name, agent.proceed_img)
                    else:
                        cv2.imshow(agent.name, conf.no_video_stream)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # Video play abort
                    cv2.destroyAllWindows()
                    break
            else:
                time.sleep(1)


if __name__ == "__main__":
    start_time = time.time()

    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard interrupt. Exit")
        exit()
    
    print(f"Program took {time.time() - start_time}s")
    print("Goodbye")