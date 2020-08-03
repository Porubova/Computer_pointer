# Computer Pointer Controller

In this project, I used a Gaze Estimation model to estimate the gaze of the user's eyes and change the
mouse pointer accordingly. I used the Inference Engine from Intel's OpenVino Toolkit and employed 
four pre-trained models
- [Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
- [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
- [Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
- [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

## The PipeLine
To coordinate the flow of the date from the input and through all four different models to the mouse controller, 
the pipeline looks like this.

![PipLine](https://github.com/Porubova/Computer_pointer/blob/master/bin/pipeline.png)

## Project Set-Up and Installation
### Prerequisites
- OpenVino 2020.4
- Python 3.7
-

- Install [OpenVino 2020.4](https://docs.openvinotoolkit.org/latest/index.html) Toolkit following the installation guide.
- Setup OpenVino Toolkit
```sh
cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
setupvars.bat
```
- clone git repository into your prefered location
```sh
git clone https://github.com/Porubova/Computer_pointer.git
```
- navigate to Computer_pointer folder

```sh
cd Computer_poiter
```
- create a virtual environment
```sh 
pip install virtualenv
virtualenv env
cd env\Scripts
activate.bat
cd ..
```
- download the required packages 
```sh
pip install -r requirements.txt
```

- Video writer (optional)
you might need to download 
Install FFmpeg as VideoCapture backend (on Windows you need to download OpenCV community plugin. 
There's downloader script in the package: openvino\opencv\ffmpeg-download.ps1.
Right click on it - Run with PowerShell).
 
## Demo


To run the demo execute the following command

```sh
cd src
python main.py -fm ..\models\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001 -hm ..\models\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001 -lm ..\models\landmarks-regression-retail-0009\FP32\landmarks-regression-retail-0009 -gm ..\models\gaze-estimation-adas-0002\FP32\gaze-estimation-adas-0002 -i ..\bin\demo.mp4 -v yes
```

## Documentation
### Project structure
```sh
.
|-- bin
|   |-- demo.mp4
|   |-- pipline.png
|-- env
|-- models
|   |-- face-detection-adas-binary-0001
|   |-- gaze-estimation-adas-0002
|   |-- head-pose-estimation-adas-0001
|   |-- landmarks-regression-retail-0009
|-- src
|   |-- face_detection.py 
|   |-- facial_landmarks_detection.py
|   |-- gaze_estimtion.py
|   |-- head_pose_estimation.py
|   |-- input_feeder.py
|   |-- main.py
|   |-- model.py
|   |-- mouse_controller.py
|-- requirements.txt
|-- README.md

```
- Folder models contains four pre-trained models folders with different precisions.
- Folder src contains python scrips 
  -  model.py is a parent class to load model and preprocess input frame
  -  mouse_controller.py responsible for importing pyautogui library, take as an input 
     mouse coords and navigate the mouse pointer accordingly
  - face_detection.py, take input frame, performs inference, detects a face, return a cropped image of the face
  - head_pose_estimation.py takes as an input cropped image of the face and returns head pose
  - facial_landmarks_detection.py takes as an input cropped image of the face and returns coordinates of the eyes
  - gaze_estimtion.py takes as input head pose and eyes coordinates, returns gaze and mouse coordinates
  - input_feeder.py responsible to feed frames from video file or camera
  - main.py takes in the arguments and call all other python scrips, records and displays the output. 
To run the program you need to run the main.py file, it takes several arguments

```sh
usage: main.py [-h] -fm FACE_MODEL -hm HEAD_MODEL -lm LANDMARKS_MODEL -gm
               GAZE_MODEL -i INPUT [-d DEVICE] [-v VISUAL]
               [-pt PROB_THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  -fm FACE_MODEL, --face_model FACE_MODEL
                        Path to an xml file with a face_detection model.
  -hm HEAD_MODEL, --head_model HEAD_MODEL
                        Path to an xml file with a head pose estimation model.
  -lm LANDMARKS_MODEL, --landmarks_model LANDMARKS_MODEL
                        Path to an xml file with a facial landmarks detection
                        model.
  -gm GAZE_MODEL, --gaze_model GAZE_MODEL
                        Path to an xml file with a head pose estimation model.
  -i INPUT, --input INPUT
                        Path to image, video file or type CAM for camera
  -d DEVICE, --device DEVICE
                        Specify the target device to infer on: CPU, GPU, FPGA
                        or MYRIAD is acceptable. Sample will look for a
                        suitable plugin for device specified (CPU by default)
  -v VISUAL, --visual VISUAL
                        Specify if you would like to display the outputs of
                        intermediate models type yes or no specified (no by
                        default)
  -pt PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Probability threshold for detections filtering(0.6 by
                        default)
```

## Benchmarks
I have tested my program using several precisions and obtained the following results:
#### FP16
- Models load time 2.74.
- Total inference time 46.00.
- Inference frames per second 1.28.
#### FP16-INT8
- Models load time 3.93.
- Total inference time 42.40.
- Inference frames per second 1.39.
#### FP32
- Models load time 2.60.
- Total inference time 41.00.
- Inference frames per second 1.44.

## Results
The FP32 precision model from visual observation performs better however it was the slowest result.
[Video](https://youtu.be/1T39af62ZiE)
![Result](https://github.com/Porubova/Computer_pointer/blob/master/bin/Capture.PNG)


### Edge Cases
The program performed reasonably well in different lighting however, it is not practical if there are several users
at the same time as it picks only one person. 
