import cv2
import logging as log
from openvino.inference_engine import IENetwork, IECore
log.basicConfig(level = log.INFO)#setting logging error

class Model:
    '''
    Parent class model.
    '''
    def __init__(self, model_name, device="CPU"):
       
        self.model_name= model_name[7:-5]
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device        
        
        try:
            log.info("Model Initiated "+self.model_name)
            self.core = IECore()
            try:
                self.model= self.core.read_network(self.model_structure, self.model_weights)
            except AttributeError:
                self.model = IENetwork(model=self.model_structure, weights=self.model_weights)
        except Exception as e:
            log.error("Error in Initialising " + self.model_name + str(e))
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")
   
    def load_model(self):
        '''
        Load the model
        '''
        try:
            log.info("Model Loaded "+self.model_name)
            
            self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)      
        except Exception as e:
            log.error("Error in loading "  + self.model_name + str(e))
            raise ValueError("Could not load the model")

    def preprocess_input(self, image):
        '''
        Pre-procees  the input frame
        '''
        try:
            p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
            p_frame = p_frame.transpose(2, 0, 1)
            p_frame = p_frame.reshape(1, *p_frame.shape)
        except Exception as e:
            log.error("Error in pre_processing the frame " + self.model_name + str(e))
            raise ValueError("Could not pre-process the image, check the input dimention")
        return p_frame

