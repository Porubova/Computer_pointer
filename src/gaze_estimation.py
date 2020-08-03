'''
The network takes three inputs: square crop of left eye image, square crop of
right eye image, and three head pose angles – (yaw, pitch, and roll) 
(see figure). The network outputs 3-D vector corresponding to the direction of
a person's gaze in a Cartesian coordinate system in which z-axis is directed
from person's eyes (mid-point between left and right eyes' centers) to the
camera center, y-axis is vertical, and x-axis is orthogonal to both z,y axes
so that (x,y,z) constitute a right-handed coordinate system.
been provided just to give you an idea of how to structure your model class.
'''
from model import Model
from math import sin, cos, pi
import logging as log
log.basicConfig(level = log.INFO)

class Gaze_Estimation_Model(Model):
    '''
    Initiate the Model.
    '''
    def __init__(self, model_name, device="CPU", threshold=0.60):
        
        self.threshold = threshold        
        Model.__init__(self, model_name, device)
        self.input_name = [key for key in self.model.inputs.keys()]
        self.input_shape = self.model.inputs[self.input_name[1]].shape
        self.output_name = [key for key in self.model.outputs.keys()]
            

    def predict(self, left_eye_image, right_eye_image, coords):
        '''
        Perform prediction
        '''
        try:
            left_eye_image = self.preprocess_input(left_eye_image)
            right_eye_image = self.preprocess_input(right_eye_image)
            input_dict = {'left_eye_image': left_eye_image,
                          'right_eye_image': right_eye_image,
                          'head_pose_angles': coords}
            self.net.start_async(request_id=0, inputs=input_dict)
            if self.net.requests[0].wait(-1) == 0:
                outputs = self.net.requests[0].outputs
                mouse_coords, gaze = self.preprocess_outputs(outputs, coords)                    
                return mouse_coords, gaze
        except Exception as e:
            log.error("Prediction error with model: "+self.model_name +" "+ str(e))
    
    def preprocess_outputs(self, outputs, coords):
        '''
       takes output
       {'gaze_vector': array([[-0.20160618, -0.41011286, -0.87488157]], dtype=float32)}
       returns gaze vector and mouse coords
        ''' 
        try:
            gaze = outputs[self.output_name[0]][0]             
            mouse_coords = (gaze[0], gaze[0])
            return mouse_coords, gaze
        except Exception as e:
            log.error("Pre-processing output error with model: "+self.model_name +" "+ str(e))
