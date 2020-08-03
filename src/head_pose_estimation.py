'''
Head pose estimation network based on simple, handmade CNN architecture.
Angle regression layers are convolutions + ReLU + batch norm + fully connected
with one output.
'''
from model import Model
import logging as log
log.basicConfig(level = log.INFO)

class Head_Pose_Model(Model):
    '''
    Class for the Head Pose Estimation Model.
    '''
    def __init__(self, model_name, device="CPU", threshold=0.60):
        '''
        Initiate model.
        
        '''
        self.threshold = threshold
        Model.__init__(self, model_name, device)
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape
        
            

    def predict(self, image):
        '''
        Perform Prediction
        '''
        try:
            p_frame = self.preprocess_input(image)
            input_dict = {self.input_name: p_frame}
            self.net.start_async(request_id=0,inputs=input_dict)
            if self.net.requests[0].wait(-1) == 0:
                outputs = self.net.requests[0].outputs
                
                pose = self.preprocess_outputs(outputs)                    
                
                return pose
        except Exception as e:
            log.error("Prediction error with model: ", self.model_name, e)
    
    def preprocess_outputs(self, outputs):
        '''
       return output
       name: "angle_y_fc", shape: [1, 1] - Estimated yaw (in degrees).
       name: "angle_p_fc", shape: [1, 1] - Estimated pitch (in degrees).
       name: "angle_r_fc", shape: [1, 1] - Estimated roll (in degrees).
        ''' 
        try:
            pose=[]
            pose.append(outputs['angle_y_fc'][0][0])
            pose.append(outputs['angle_p_fc'][0][0])
            pose.append(outputs['angle_r_fc'][0][0])
            
            
            
            
            return pose
        except Exception as e:
            log.error("Pre-processing output error with model: ", self.model_name, e)
    
