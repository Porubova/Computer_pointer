from argparse import ArgumentParser
import cv2
import time
from face_detection import Face_Detection_Model
from head_pose_estimation import Head_Pose_Model
from facial_landmarks_detection import Facial_Landmarks_Model
from gaze_estimation import Gaze_Estimation_Model
from input_feeder import InputFeeder
from mouse_controller import MouseController
import numpy as np
import math
import logging as log

log.basicConfig(level = log.INFO)
def get_args():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fm","--face_model",  required=True, type=str,
                        help="Path to an xml file with a face_detection model.")
    parser.add_argument("-hm","--head_model",  required=True, type=str,
                        help="Path to an xml file with a head pose estimation model.")
    parser.add_argument("-lm","--landmarks_model",  required=True, type=str,
                        help="Path to an xml file with a facial landmarks detection model.")
    parser.add_argument("-gm","--gaze_model",  required=True, type=str,
                        help="Path to an xml file with a head pose estimation model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image,  video file or type CAM for camera")
    
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-v", "--visual", type=str, default="no",
                        help="Specify if you would like to "
                            "display the outputs of intermediate models "
                             "type yes or no "
                             "specified (no by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.6,
                        help="Probability threshold for detections filtering"
                        "(0.6 by default)")    
    
    return parser
    

def infer_on_stream(args):
    
    start_model_load_time=time.time()
    
    #initiate and load models
    face_det_net = Face_Detection_Model(args.face_model)
    face_det_net.load_model()
    head_pose_net = Head_Pose_Model(args.head_model)
    head_pose_net.load_model()
    facial_landmarks_net = Facial_Landmarks_Model(args.landmarks_model)
    facial_landmarks_net.load_model()
    gaze_est_net = Gaze_Estimation_Model(args.gaze_model)
    gaze_est_net.load_model()
    total_model_load_time = time.time() - start_model_load_time
    
    #initiate stream
    counter=0
    start_inference_time=time.time()
    
    if args.input.lower()=="cam":
        frame_feeder = InputFeeder(input_type='cam')
        frame_feeder.load_data()
    else:
        frame_feeder = InputFeeder(input_type='video', input_file=args.input)
        frame_feeder.load_data()
    fps = frame_feeder.get_fps()
    log.info('Video started')
    
    #initiate mouse controller
    mouse_controller = MouseController('medium','fast')
    
    ## write output video in Winows
    out_video = cv2.VideoWriter('../output.mp4',cv2.VideoWriter_fourcc(*'avc1'),
                                fps,(frame_feeder.get_size()), True)
    
    ## write output video in Linux
    #out_video = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'avc1'),
    #fps,(frame_feeder.get_size()))
    
    for flag,frame in frame_feeder.next_batch():
        if flag == True:             
            key = cv2.waitKey(60) 
            counter+=1
            coords, image, face = face_det_net.predict(frame)
            pose = head_pose_net.predict(face)
            land, left_eye_image, right_eye_image, eye_coords = facial_landmarks_net.predict(face)
            
            if left_eye_image.shape == (40, 40, 3):
                mouse_coords, gaze = gaze_est_net.predict(left_eye_image, right_eye_image, pose)
                
            mouse_controller.move(mouse_coords[0], mouse_coords[1])
            
            if args.visual.lower()=="yes":
                frame = draw_outputs(coords, eye_coords, pose, gaze, 
                                     mouse_coords[0], mouse_coords[1],
                                     image)
                cv2.imshow('video', frame)
                out_video.write(frame)
                cv2.imshow('video', frame)
            else:
                cv2.imshow('video', frame)
            if key == 27:
                break 
        else:
            log.info('Video ended')
            total_time=time.time()-start_inference_time
            total_inference_time=round(total_time, 1)
            f_ps=counter/total_inference_time
            log.info("Models load time {:.2f}.".format(total_model_load_time))
            log.info("Total inference time {:.2f}.".format(total_inference_time))
            log.info("Inference frames pre second {:.2f}.".format(f_ps))
            cv2.destroyAllWindows()
            frame_feeder.close()
            break
    
def draw_outputs(coords, eye_coords, pose, gaze,x, y, image):
    '''
    Display theintermedium prezentation of models
    '''
    
    face_center = (int((coords[0][2]+coords[0][0])/2),
                   int((coords[0][1]+coords[0][3])/2))
    left_eye_center = (int((coords[0][0]+eye_coords[0][2]+coords[0][0]+eye_coords[0][0])/2),
                   int((coords[0][1]+eye_coords[0][1]+coords[0][1]+eye_coords[0][3])/2))
   
    
    right_eye_center = (int((coords[0][0]+eye_coords[1][2]+coords[0][0]+eye_coords[1][0])/2),
                   int((coords[0][1]+eye_coords[1][1]+coords[0][1]+eye_coords[1][3])/2))
    cv2.rectangle(image, (coords[0][0], coords[0][1]), (coords[0][2], coords[0][3]), (0, 255,0), 1)
    
    
    
#    cv2.rectangle(image, (coords[0][0]+eye_coords[0][0], coords[0][1]+eye_coords[0][1]),
#                  (coords[0][0]+eye_coords[0][2], coords[0][1]+eye_coords[0][3]),
#    (0, 255,0), 1)
#    cv2.rectangle(image, (coords[0][0]+eye_coords[1][0], coords[0][1]+eye_coords[1][1]),
#                  (coords[0][0]+eye_coords[1][2], coords[0][1]+eye_coords[1][3]),
#    (0, 255,0), 1)
    
#    cv2.circle(image, (face_center), radius=1, color=(0, 255,0), thickness=2)
    
    cv2.circle(image, (left_eye_center), radius=10, color=(0, 255,0), thickness=1)
    cv2.circle(image, (right_eye_center), radius=10, color=(0, 255,0), thickness=1)
    image = draw_axes(image, face_center, pose[0], pose[1], pose[2], 50, 950.0)
    image = draw_axes(image, left_eye_center, gaze[2], gaze[1], gaze[0]*100, 150, 950.0)
    image = draw_axes(image, right_eye_center, gaze[2], gaze[1], gaze[0]*100, 150, 950.0)
  
    return image

def draw_axes(frame, center_of_face, yaw, pitch, roll, scale, focal_length):
    
    #credits to
    #https://knowledge.udacity.com/questions/171017
    
    yaw *= np.pi / 180.0
    pitch *= np.pi / 180.0
    roll *= np.pi / 180.0
    cx = int(center_of_face[0])
    cy = int(center_of_face[1])
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(pitch), -math.sin(pitch)],
                   [0, math.sin(pitch), math.cos(pitch)]])
    Ry = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                   [0, 1, 0],
                   [math.sin(yaw), 0, math.cos(yaw)]])
    Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                   [math.sin(roll), math.cos(roll), 0],
                   [0, 0, 1]])
    # R = np.dot(Rz, Ry, Rx)
    # ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    # R = np.dot(Rz, np.dot(Ry, Rx))
    R = Rz @ Ry @ Rx
    # print(R)
    camera_matrix = build_camera_matrix(center_of_face, focal_length)
    xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
    yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
    zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
    zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)
    o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
    o[2] = camera_matrix[0][0]
    xaxis = np.dot(R, xaxis) + o
    yaxis = np.dot(R, yaxis) + o
    zaxis = np.dot(R, zaxis) + o
    zaxis1 = np.dot(R, zaxis1) + o
    xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 0, 255), 1)
    xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 255, 0), 1)
    xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
    yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
    p1 = (int(xp1), int(yp1))
    xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (255, 0, 0), 2)
    #cv2.circle(frame, p2, 3, (255, 0, 0), 2)
    return frame

def build_camera_matrix(center_of_face, focal_length):
    cx = int(center_of_face[0])
    
    cy = int(center_of_face[1])
    camera_matrix = np.zeros((3, 3), dtype='float32')
    camera_matrix[0][0] = focal_length
    camera_matrix[0][2] = cx
    camera_matrix[1][1] = focal_length
    camera_matrix[1][2] = cy
    camera_matrix[2][2] = 1
    return camera_matrix
        
    
def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = get_args().parse_args()
    
    # Perform inference on the input 
    infer_on_stream(args)


if __name__ == '__main__':
    main()