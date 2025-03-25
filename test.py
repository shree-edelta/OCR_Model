import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
import cv2 as cv
# import changes as ch
import matplotlib.pyplot as plt

def resize_images(img, target_height, target_width):
    resized_images = []
    img_resized = tf.image.resize(img, (target_height, target_width))
    img_resized = tf.image.convert_image_dtype(img_resized, tf.float32)
    img_resized /= 255.0
    resized_images.append(img_resized)
    return np.array(resized_images)

def load_image(image_array, target_size=(32, 128)):
    img = image_array.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize
    return img_array


loaded_model = tf.keras.models.load_model('cnn_rnn_model.h5', compile=False)


# test_images, test_labels = ch.convert_data('dataset/test_process_data.csv')

# max_label_length = max(len(label) for label in test_labels)
# test_images_resized = ch.resize_images(test_images, 32, 128) 
# test_labels_padded = ch.pad_sequences(test_labels, maxlen=max_label_length, padding='post', value=0)  


# test_loss = loaded_model.evaluate(test_images_resized, test_labels_padded)
# print(f'Test Loss: {test_loss}')

# # predictions on test data
# y_pred = loaded_model.predict(test_images_resized)
# print(f"Predicted shape: {y_pred.shape}")

# img  =  "hform.jpg"
img = cv.imread('images/hf.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_r = cv.resize(img_gray, (220, 55))


cv.imshow("img", img)
cv.waitKey(0)

cv.imshow("img_gray", img_r)
cv.waitKey(0)

cv.destroyAllWindows()

print(img.shape)
print(img_r.shape)
# img_resize = resize_images(img, 32, 128)
# print(img_l)
# # images_array = np.expand_dims(img_l, axis=-1) 
# predictions = loaded_model.predict(img_l)
# print(predictions)

# predicted_class = np.argmax(predictions, axis=-1)
# print(f"Predicted class: {predicted_class}")
