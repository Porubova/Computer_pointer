'''
Face detection model.
Face detector for driver monitoring and similar scenarios. The network features
a pruned MobileNet backbone that includes depth-wise convolutions to reduce 
the amount of computation for the 3x3 convolution block. Also some 1x1
convolutions are binary that can be implemented using effective binary 
XNOR+POPCOUNT approach
'''

from model import Model
import cv2
import logging as log
log.basicConfig(level = log.INFO)

class Face_Detection_Model(Model):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device="CPU", threshold=0.60):
        '''
        Initiate the model.
        
        '''
        
        self.threshold = threshold
        Model.__init__(self, model_name, device)
        self.input_name=next(iter(self.model.input_info))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape
        
            

    def predict(self, image):
        '''
        Perform prediction.
        '''
        try:
            p_frame = self.preprocess_input(image)
            input_dict = {self.input_name: p_frame}
            self.net.start_async(request_id=0,inputs=input_dict)
            if self.net.requests[0].wait(-1) == 0:
                outputs = self.net.requests[0].outputs[self.output_name]
                boxes, image, face_image = self.preprocess_outputs(outputs, image)           
                
                return boxes, image, face_image
        except Exception as e:
            log.error("Prediction error with model: "+self.model_name +" "+ str(e))
    
    def preprocess_outputs(self, outputs, image):
        '''
        Input:
        The net outputs blob with shape: [1, 1, N, 7], where N is the number of
        detected bounding boxes. Each detection has the format 
        [image_id, label, conf, x_min, y_min, x_max, y_max], where:

        image_id - ID of the image in the batch
        label - predicted class ID
        conf - confidence for the predicted class
        (x_min, y_min) - coordinates of the top left bounding box corner
        (x_max, y_max) - coordinates of the bottom right bounding box corner.
        Output:
        coords of the cropped face and coords of the face detected.
        
        ''' 
        try:            
            face_image = image
            boxes = []
            coords = []
            for box in outputs[0][0]:
                conf = box[2]
                if conf >= self.threshold:
                    coords.append(box)
            for box in coords:
                xmin = int(box[3] * image.shape[1])
                ymin = int(box[4] * image.shape[0])
                xmax = int(box[5] * image.shape[1])
                ymax = int(box[6] * image.shape[0])
                #cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255,0), 1)
                boxes.append([xmin, ymin, xmax, ymax])
                face_image = image[ymin:ymax, xmin:xmax]
            return boxes, image, face_image
        except Exception as e:
            log.error("Pre-processing output error with model: "+self.model_name +" "+ str(e))
