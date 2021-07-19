import cv2
from tensorflow import keras
import numpy as np


# load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# load IR model and specify NCS
model_path = "converted_model/saved_model.xml"
arch_path = "converted_model/saved_model.bin"
net = cv2.dnn.readNet(arch_path, model_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# use these data to test
images = x_train.reshape(60000, 28 * 28)/255
images = images.astype("float32")
labels = y_train
correct = 0

# test all of them
for i in range(10000):
    img = images[i]
    net.setInput(img)
    out = net.forward()
    if np.argmax(out, axis = 1) == labels[i]:
        correct += 1

print(correct/10000)
# path_to_images = "gpu_test/notebooks/"
# img = plt.imread(path_to_images + "6.jpg")
# img = img[:, :, 0]
# img = cv2.resize(img, (28, 28))
# img = img.reshape(1, 28 * 28)/255
# img = img.astype("float32")

# net.setInput(img)
# out = net.forward()
# print(np.argmax(out, axis = 1))