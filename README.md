# Mask-Detector
This little project should detect all persons in an image/video and mark everybody red (unmasked) and blue (masked). It can help stores to check if their clients wear masks during the corona-pandemic!


## Arguments
You have to give at least one argument to run this code. There are some available:
- `-help` That is clear
- `-test` Test images inside 'data/test' and returns the result per image
- `-webcam` Get a videostream of a webcam and check for faces
- `-train [epochs]` Train images. Epochs has to be given



## How can I train the model?
You have to make three additionel directories inside 'data/train/':
- no_mask     (Images without masks)
- tensors     (Leave it empty)
- with_mask   (Images with masks)

If you have all your images prepared and inserted, simply run `python3 main.py -newbatches -train 5`. This will start the program and make first some training batch files and then trains over 5 epochs. The early created batch files help you to not calculate every time new tensors (Efficiency aspects).
