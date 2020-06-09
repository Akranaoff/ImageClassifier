
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
#from cnn_classifier import  train_val

class friend:
    def __init__(self,filename):
        self.filename = filename

    def predictor(self):
        model = load_model("multi_class_model.h5")
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (64,64))   #loading your image to be classified
        test_image = image.img_to_array(test_image)                     #converting the testing image to array test_image
        test_image = np.expand_dims(test_image, axis=0)                 #it is used to expand the shape row or columns wise ie to convert [2,3] to [[2,3]]
        result = model.predict(test_image)
        loc1,loc2 = np.where(result == 1.)
        train_val = {'Akshay': 0, 'Kedar': 1, 'Lassi': 2, 'Saurabh': 3}
        for i,j in train_val.items():
            if j == loc2:
                predict = i
                return [{'image' : predict}]
