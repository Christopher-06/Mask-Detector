from network import *
import config as conf
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('-train', help='Start a training session. Specify the epochs', type=int)
parser.add_argument('-test', help='Test all images and get result', action='store_true')
parser.add_argument('-webcam', help='Streaming a webcam', action='store_true')
parser.add_argument('-newbatches', help='make new batch files', action='store_true')
args = parser.parse_args()

def main():
    start_time = time.time()
    model = load()    

    if args.newbatches:
        make_training_batch_files()
    if args.train:
        model = Trainer(model, args.train).run()
    if args.test:
        Tester(model).run()
    if args.webcam:
        Webcam_Agent(model).run()


    print(f"Program took {time.time() - start_time}s")
    print("Goodbye")

if __name__ == "__main__":
    main()