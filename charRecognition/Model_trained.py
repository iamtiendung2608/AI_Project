from Model_class import CNN_Model
import cv2
import numpy as np
from preprocess import *
model = CNN_Model()
model.train()
model.save('my_model.h5')
