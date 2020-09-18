# Mask-Detector
This little project should detect all persons in an image/video and mark everybody red (unmasked) and blue (masked). It can help stores to check if their clients wear masks during the corona-pandemic!

| [![Unmasked Person](https://d1ow200m9i3wyh.cloudfront.net/img/assets/videoblocks/images/logo.png)](http://videoblocks.com)  | [![Masked Person](https://raw.githubusercontent.com/Christopher-06/Mask-Detector/master/data/test/with_mask/example.jpg)](http://audioblocks.com) | [![GraphicStock](http://www.graphicstock.com/images/logo.jpg)](http://graphicstock.com) |
|:---:|:---:|:---:|
| http://videoblocks.com | http://audioblocks.com | http://graphicstock.com |
## Arguments
You have to give at least one argument to run this code. There are some available:
- `-help` That is clear
- `-test` Test images inside 'data/test' and returns the result per image
- `-webcam` Get a videostream of a webcam and check for faces
- `-train [epochs]` Train images. Epochs has to be given
- `-video` Test and/or train entire videos



## How to use?
First you have to define a webcam in the webcam.json file. If you don't own a webcam, you can simply download an IP Webcam app. I am using this one: [IP Webcam (Android)](https://play.google.com/store/apps/details?id=com.pas.webcam)

After downloading, you can start it and press the button 'Start server' at the bottom. On your screen should something be visible like `http://192.168.178.38:8080`. Enter this link in a browser and you see the livestream of your mobile camera. Then open the webcam.json file. Paste the link and add `/video`. The reason is that we only want the videostream. At the end you can name it as you want.

`[ { "name" : "mobile_camera", "url" : "paste_here_your_link + /video" } ... ]`

You can add more webcams easily: Copy and paste this `,{ "name" : "mobile_camera", "url" : "paste_here_your_link + /video" }`. NOTE: You have to give each cam another name. Otherwise there will be only on window.

If you done everything correctly you can double click `Launch Webcam` or run `python main.py -webcam` in a cmd.
