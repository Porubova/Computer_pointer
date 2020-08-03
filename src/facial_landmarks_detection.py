'''
This is a lightweight landmarks regressor for the Smart Classroom scenario.
It has a classic convolutional design: stacked 3x3 convolutions, 
batch normalizations, PReLU activations, and poolings. 
Final regression is done by the global depthwise pooling head and 
FullyConnected layers. 
The model predicts five facial landmarks: two eyes, nose, and two lip corners.
'''
from model import Model
import logging as log
log.basicConfig(level = log.INFO)
class Facial_Landmarks_Model(Model):
    '''
    Class for the Facial Landmark detection Model.
    '''
    def __init__(self, model_name, device="CPU", threshold=0.60):
        '''
        Initiate the model.
        
        '''
        self.threshold = threshold
        Model.__init__(self, model_name, device)
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape       
            

    def predict(self, image):
        '''
        Perform prediction
        '''
        try:
            p_frame = self.preprocess_input(image)
            input_dict = {self.input_name: p_frame}
            self.net.start_async(request_id=0,inputs=input_dict)
            if self.net.requests[0].wait(-1) == 0:
                outputs = self.net.requests[0].outputs[self.output_name]
    
                image, left_eye_image, right_eye_image, eye_coords = self.preprocess_outputs(outputs, image)                        
                return image, left_eye_image, right_eye_image, eye_coords
        except Exception as e:
            log.error("Prediction error with model: "+self.model_name +" "+ str(e))
    
    def preprocess_outputs(self, outputs, image):
        '''
        Takes net outputs a blob with the shape: [1, 10], 
        containing a row-vector of 10  floating point values for five landmarks 
        coordinates in the form (x0, y0, x1, y1, ..., x5, y5).
        All the coordinates are normalized to be in range [0,1].
        Returns coordinates for left and righ eyes
        ''' 
        try:
            output = outputs[0]
            
            left_eye_xmin = int(output[0][0][0] * image.shape[1]) - 20
            left_eye_ymin = int(output[1][0][0] * image.shape[0]) - 20
            right_eye_xmin = int(output[2][0][0] * image.shape[1]) - 20
            right_eye_ymin = int(output[3][0][0] * image.shape[0]) - 20
    
            left_eye_xmax = int(output[0][0][0] * image.shape[1]) + 20
            left_eye_ymax = int(output[1][0][0] * image.shape[0]) + 20
            right_eye_xmax = int(output[2][0][0] * image.shape[1]) + 20
            right_eye_ymax = int(output[3][0][0] * image.shape[0]) + 20
            
    #        cv2.rectangle(image, (left_eye_xmin, left_eye_ymin), (left_eye_xmax, left_eye_ymax), (0, 255,0), 1)
    #        cv2.rectangle(image, (right_eye_xmin, right_eye_ymin), (right_eye_xmax, right_eye_ymax), (0, 255,0), 1)
            left_eye_image = image[left_eye_ymin:left_eye_ymax, left_eye_xmin:left_eye_xmax]
            right_eye_image = image[right_eye_ymin:right_eye_ymax, right_eye_xmin:right_eye_xmax]
    
            eye_cords = [[left_eye_xmin, left_eye_ymin, left_eye_xmax, left_eye_ymax],
                             [right_eye_xmin, right_eye_ymin, right_eye_xmax, right_eye_ymax]]
            
            return image, left_eye_image, right_eye_image, eye_cords
        
        except Exception as e:
            log.error("Pre-processing output error with model: "+self.model_name +" "+ str(e))