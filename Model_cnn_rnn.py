import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from keras.utils import to_categorical
from tensorflow.keras import backend as K
import numpy as np

# df_train = pd.read_csv('train_process_data.csv')
# train_images = df_train['images']
# train_labels = df_train['labels']

# df_val = pd.read_csv('val_process_data.csv')
# val_images = df_val['images']
# val_labels = df_val['labels']

df_test = pd.read_csv('dataset/written_name_train_v2.csv')
test_images = "dataset/test_v2/test" + df_test['FILENAME']
test_labels = df_test['IDENTITY']

# def convert_data(file_name):   
#     data = pd.read_csv(file_name)
#     # data = data.dropna()
   
#     f_array=[]
#     for i in range(len(data['images'])):
#         array_str = data['images'][i]  
#         str_list = data['labels'][i]
        
#         str_list = str_list.strip().strip('[]').split(',') 
#         int_list = list(map(int, str_list))
        
#         array_str_cleaned = array_str.replace('[', '').replace(']', '').replace('\n', ' ')
#         array_list = np.fromstring(array_str_cleaned, sep=' ')
#         array_list = array_list.reshape((3,3))
#         array = np.array(array_list)
#         f_array.append(array)
#     return f_array,int_list


# def convert_data(file_name):   
#     data = pd.read_csv(file_name)
   
#     f_array = []
#     label_list = []  # To store labels as sequences
    
#     for i in range(len(data['images'])):
#         array_str = data['images'][i]  
#         str_list = data['labels'][i]
        
#         # Convert label string to list of integers
#         str_list = str_list.strip().strip('[]').split(',') 
#         int_list = list(map(int, str_list))
        
#         # Process image string into a numpy array (reshape and convert)
#         array_str_cleaned = array_str.replace('[', '').replace(']', '').replace('\n', ' ')
#         array_list = np.fromstring(array_str_cleaned, sep=' ')
#         array_list = array_list.reshape((3, 3))  # Adjust the reshape based on your input image dimensions
#         array = np.array(array_list)
#         f_array.append(array)
#         label_list.append(int_list)

 # # Convert image data to numpy array
    # images_array = np.array(f_array)

    # # Return images and labels as lists (or padded numpy arrays if needed)
    # return images_array, label_list  # Return label_list as a list of lists

def convert_data(file_name):   
    data = pd.read_csv(file_name)
   
    f_array = []
    label_list = []  # To store labels as sequences
    
    for i in range(len(data['images'])):
        array_str = data['images'][i]  
        str_list = data['labels'][i]
        
        # Convert label string to list of integers
        str_list = str_list.strip().strip('[]').split(',') 
        int_list = list(map(int, str_list))
        
        # Process image string into a numpy array (reshape and convert)
        array_str_cleaned = array_str.replace('[', '').replace(']', '').replace('\n', ' ')
        array_list = np.fromstring(array_str_cleaned, sep=' ')
        array_list = array_list.reshape((3, 3))  # Adjust the reshape based on your input image dimensions
        array = np.array(array_list)
        f_array.append(array)
        label_list.append(int_list)
    
    # Convert image data to numpy array
    images_array = np.array(f_array)
    
    # Normalize image data (if needed)
    images_array = images_array.astype('float32') / 255.0  # Normalize to [0, 1]
    
    # Add a channel dimension (for grayscale images)
    images_array = np.expand_dims(images_array, axis=-1)  # Convert from (num_samples, height, width, 3) to (num_samples, height, width, 1)

    # Return images and labels as lists
    return images_array, label_list

    
   
# train_images,train_labels= convert_data('train_process_data.csv')

# val_images,val_labels = convert_data('val_process_data.csv')

# def build_handwriting_recognition_model(img_height, img_width, num_classes):
#     # CNN layers for feature extraction
#     input_img = layers.Input(shape=(img_height, img_width, 1))  

#     # CNN layers
#     x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)
#     x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)
#     x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)

#     # Reshape the feature maps to be compatible with RNN
#     new_shape = (x.shape[1], x.shape[2] * x.shape[3])  
#     x = layers.Reshape(target_shape=new_shape)(x)

#     # RNN layers (LSTM)
#     x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    
#     # Output layer
#     output = layers.Dense(num_classes, activation='softmax')(x)

#     model = models.Model(inputs=input_img, outputs=output)

#     return model


def build_handwriting_recognition_model(img_height, img_width, num_classes):
    # CNN layers for feature extraction
    input_img = layers.Input(shape=(img_height, img_width, 1))  # 1 channel for grayscale images

    # CNN layers
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Reshape the feature maps to be compatible with RNN
    new_shape = (x.shape[1], x.shape[2] * x.shape[3])  
    x = layers.Reshape(target_shape=new_shape)(x)

    # RNN layers (LSTM)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    
    # Output layer
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=input_img, outputs=output)

    return model




def ctc_loss(y_true, y_pred):
    input_length = tf.ones_like(y_true, dtype=tf.int32) * tf.shape(y_pred)[1]
    label_length = K.sum(K.cast(K.not_equal(y_true, 0), dtype=K.int32), axis=-1)
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

img_height, img_width = 32, 128  
num_classes = 27  
img_channels = 3



train_images, train_labels = convert_data('train_process_data.csv')
val_images, val_labels = convert_data('val_process_data.csv')

# print("Shape of train_images:", train_images.shape)
# print("Shape of train_labels:", len(train_labels))



def resize_images(images, target_height, target_width):
    resized_images = []
    for img in images:
        img_resized = tf.image.resize(img, (target_height, target_width))
        resized_images.append(img_resized)
    return np.array(resized_images)

# Example: resize all images to 32x128
train_images_resized = resize_images(train_images, 32, 128)
val_images_resized = resize_images(val_images, 32, 128)
# print("train image size:",train_images.shape)
# train_images_resized = np.squeeze(train_images_resized, axis=-1)
# val_images_resized = np.squeeze(val_images_resized, axis=-1)

# train_images = np.expand_dims(train_images_resized, axis=-1)  # This will make the shape (330961, 32, 128, 1)
# val_images = np.expand_dims(val_images_resized, axis=-1)

train_images_resized = train_images_resized.astype('float32') 
val_images_resized = val_images_resized.astype('float32') 

train_images_resized /= 255.0
val_images_resized /= 255.0

train_labels_flattened = [item for sublist in train_labels for item in sublist]
train_labels = np.array(train_labels_flattened)
print(f"Shape of train_labels: {train_labels.shape}")
print(train_labels[:10])

val_labels_flattened = [item for sublist in val_labels for item in sublist]
val_labels = np.array(val_labels_flattened)
print(f"Shape of val_labels: {val_labels.shape}")
print(val_labels[:10])

# Print the shapes and data types
# val_labels = np.array(val_labels)
# val_labels = val_labels.flatten()

max_label = np.max(train_labels)
num_classes = max_label + 1

max_label_v = np.max(val_labels)
num_classes_v = max_label_v + 1

train_labels = to_categorical(train_labels, num_classes=num_classes)  # Replace num_classes with the correct number of classes
val_labels = to_categorical(val_labels, num_classes=num_classes_v)

train_labels = train_labels[:train_images_resized.shape[0]]
val_labels = val_labels[:val_images_resized.shape[0]]

print(f"Train images shape: {train_images_resized.shape}, type: {train_images_resized.dtype}")
print(f"Train labels shape: {train_labels.shape}, type: {train_labels.dtype}")
print(f"Validation images shape: {val_images_resized.shape}, type: {val_images_resized.dtype}")
print(f"Validation labels shape: {val_labels.shape}, type: {val_labels.dtype}")


model = build_handwriting_recognition_model(img_height, img_width, num_classes)
model.compile(optimizer=Adam(), loss=ctc_loss)
model.fit(train_images_resized, train_labels, epochs=10, batch_size=32, validation_data=(val_images_resized, val_labels))
model.summary()

# Define model
# img_height, img_width = 32, 128  
# num_classes = 27  
# img_channels = 3
# model = build_handwriting_recognition_model(img_height, img_width, num_classes)

# model.compile(optimizer=Adam(), loss=ctc_loss)

# model.fit(train_images, train_labels, epochs=10, batch_size=32,validation_data=(val_images, val_labels))

# model.summary()
# model.save('cnn_rnn_model.h5')

# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print(f'Test Accuracy: {test_acc}')


# y_pred = model.predict(test_images)
# print(y_pred.shape)

