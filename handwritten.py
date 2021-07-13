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


print("should print 2, if works move on")
img = plt.imread("test_image.jpg")
img = img[:, :, 1]
img = cv2.resize(img, (1, 28 * 28))


net.setInput(img)
out = net.forward()
print(np.argmax(out, axis=1))



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