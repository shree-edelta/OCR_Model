# from datasets import load_dataset

# ds = load_dataset("priyank-m/MJSynth_text_recognition")

# print(ds)

# import requests
# headers = {"Authorization": f"Bearer {API_TOKEN}"}
# API_URL = "https://datasets-server.huggingface.co/parquet?dataset=ibm/duorc"
# def query():
#     response = requests.get(API_URL, headers=headers)
#     return response.json()
# data = query()

# print(data)

# import requests
# headers = {"Authorization": f"Bearer {API_TOKEN}"}
# API_URL = "https://huggingface.co/api/datasets/ibm/duorc/parquet"
# def query():
#     response = requests.get(API_URL, headers=headers)
#     return response.json()
# urls = query()

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Function to remove extensions and get the base name
def remove_extensions(file_name):
    base_name = os.path.splitext(file_name)[0]
    # for .gt file
    while '.' in base_name:
        base_name = os.path.splitext(base_name)[0]
    return base_name


folder_path = 'archive 2/Template1/training_data'  

all_files = os.listdir(folder_path)

tiff_files = []
gt_txt_files = []
labels =[]
for file in all_files:
    if file.endswith('.tiff'):
        tiff_files.append(file)
    elif file.endswith('.gt.txt'):
        gt_txt_files.append(file)
# print(tiff_files)
for tiff_file in tiff_files:
    tiff_base = remove_extensions(tiff_file)
    # print(tiff_base)
    matching_txt_file = None
    
    for txt_file in gt_txt_files:
        if remove_extensions(txt_file) == tiff_base:
            matching_txt_file = txt_file
            # print(matching_txt_file)
            break

    if matching_txt_file:
        with open(os.path.join(folder_path, matching_txt_file), 'r') as file:
            label = file.read().strip()
            labels.append(label)
        # print(f"File: {tiff_file} has label: {label}")
    else:
        print(f"No matching label found for {tiff_file}")

def load_image(image_path, target_size=(128, 128)):
    image = Image.open(image_path).convert('RGB') 
    image = image.resize(target_size)  
    image = np.array(image)  
    image = image / 255.0  
    return image

images = []
# print(tiff_files)
for tiff_file in tiff_files:
    # print(tiff_file)
    base_name = remove_extensions(tiff_file)
    matching_txt_file = base_name + '.gt.txt'
    # print(matching_txt_file)
    # Only load image if there is a matching label
    if matching_txt_file in all_files:
        img_path = os.path.join(folder_path, tiff_file)
        image = load_image(img_path)
        images.append(image)

# print(images)
X = np.array(images) 
# y = np.array(labels)  
# print(len(X))
# print(len(y))

# # print(y)
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)
# print("encoder......",y_encoded)
# y_one_hot = tf.keras.utils.to_categorical(y_encoded)  
# print("hot......>>>>>>>>>>>>>>>>",y_one_hot)

label_encoder = LabelEncoder()
all_characters = set(''.join(labels))
label_encoder.fit(sorted(all_characters))
y_encoded = [label_encoder.transform(list(label)) for label in labels]

max_label_length = max(len(label) for label in y_encoded)
y_encoded = [label + [0] * (max_label_length - len(label)) for label in y_encoded]
y = np.array(y_encoded)
# y_encoded = label_encoder.fit_transform(y)
# print(len(y_encoded))
# print("encoder......", y_encoded)
# num_classes = len(label_encoder.classes_)
# print(num_classes)
# y_one_hot = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)

# # y_one_hot = tf.one_hot(y_encoded, depth=np.max(y_encoded) + 1)
# print("hot......>>>>>>>>>>>>>>>>", y_one_hot)
# num_classes = np.max(y_encoded) + 1


# y_one_hot = np.eye(num_classes)[y_encoded]

# print("One-hot encoded labels:\n", y_one_hot)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
num_classes = len(label_encoder.classes_)
# #  CNN model
# model = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),  
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(128, (3, 3), activation='relu'),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(y_one_hot.shape[1], activation='softmax') 
# ])


# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))


# # model.save('cnn_model.h5')

# test_loss, test_acc = model.evaluate(X_test, y_test)
# print(f'Test Accuracy: {test_acc}')

# num_classes = len(y) 
# print(num_classes)
model = models.Sequential([
  
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
   
    layers.Reshape((-1, 128)), 
    
    layers.LSTM(128, return_sequences=True, dropout=0.25),
    layers.Dropout(0.25),

    layers.Dense(num_classes, activation='softmax')
])

# label_lengths = np.array([len(label) for label in y_true])
# def ctc_loss(y_true, y_pred):
#     print(y_true.shape)
#     label_lengths = tf.fill([tf.shape(y_true)[0]], 10)  # Adjust this based on the actual label lengths
#     logit_lengths = tf.fill([tf.shape(y_pred)[0]],  tf.shape(y_pred)[1]) 
#     print(">>>>>>>>>>>>>>>>>>>>>>>>>>",y_pred.shape)
#     y_true = tf.cast(y_true, tf.int32)
#     return tf.reduce_mean(tf.nn.ctc_loss(labels=y_true, logits=y_pred, label_length=label_lengths, logit_length=logit_lengths))

# def ctc_loss(y_true, y_pred):
#     # print(f"y_true shape: {y_true.shape}")
#     # print(f"y_pred shape: {y_pred.shape}")
#     # y_true = tf.cast(y_true, tf.int32)
#     # y_true = y_true[:, :max_label_length] 
    
#     y_true = tf.squeeze(y_true, axis=0)  # Remove the leading dimension
#     y_true = tf.cast(y_true, tf.int32)
#     y_pred = tf.squeeze(y_pred, axis=-2)  # Remove the extra singleton dimension

#     # y_pred = tf.squeeze(y_pred, axis=-2)
#     # # y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1, num_classes])
#     # print(type(y_true))
#     label_lengths = tf.fill([tf.shape(y_true)[0]], 139)  # Length of each label sequence (e.g., 139)
#     logit_lengths = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
#     # label_lengths = tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), tf.int32), axis=1)
    
    
#     # logit_lengths = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])  
    
#     return tf.reduce_mean(tf.nn.ctc_loss(labels=y_true, logits=y_pred, label_length=label_lengths, logit_length=logit_lengths))


# ///////////////////////////////////////////////

def ctc_loss(y_true, y_pred):
    # Ensure y_true is of type int32 (labels should be integers)
    y_true = tf.cast(y_true, tf.int32)

    # Reshape y_true to [batch_size, max_label_length] if it's not already
    y_true = tf.squeeze(y_true, axis=-1)  # Removing unnecessary dimensions if needed

    # Calculate label lengths dynamically (number of characters in each label)
    label_lengths = tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), tf.int32), axis=1)  # Find length of each label (ignoring padding)

    # Reshape y_pred to [batch_size, time_steps, num_classes]
    y_pred = tf.squeeze(y_pred, axis=-2)  # Remove the singleton dimension (1) from the shape if it's unnecessary

    # Calculate logit lengths dynamically (number of time steps in each prediction)
    logit_lengths = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])  # Number of time steps for each prediction

    # Compute CTC loss
    return tf.reduce_mean(tf.nn.ctc_loss(labels=y_true, logits=y_pred, label_length=label_lengths, logit_length=logit_lengths))

# Example of how you might use this function in a model
# model.compile(optimizer='adam', loss=ctc_loss)


model.compile(optimizer='adam', loss=ctc_loss)

model.summary()

model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

# model.save('lstm_model.h5')

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc}')

# y_pred = model.predict(X_test)
# print(y_pred.shape)
