import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from openvino.inference_engine import IENetwork, IECore

model_path = "converted_model/saved_model.xml"
arch_path = "converted_model/saved_model.bin"

net = cv2.dnn.readNet(arch_path, model_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)


vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()

    dat = frame[:, :, 1]
    dat = cv2.resize(dat, (1, 28 * 28))

    net.setInput(dat)
    out = net.forward()
    
    num = np.argmax(out, axis = 1)
    print(num)
    
    cv2.imshow('Input', frame)

    # Press "ESC" key to stop webcam
    if cv2.waitKey(1) == 27:
        break


# Release video capture object and close the window
vid.release()
cv2.destroyAllWindows()
cv2.waitKey(1)