# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:28:46 2020

@author: Tech Land
"""


from keras.models import model_from_json
import numpy as np

class PersonDetectModel(object):

    PERSON_LIST = ["forhad", "rahat",
                     "rifat", "riyadh"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        print("Model loaded from disk")
        self.loaded_model.summary()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)

        return PersonDetectModel.PERSON_LIST[np.argmax(self.preds)]


if __name__ == '__main__':
    pass