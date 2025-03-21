import pandas as pd
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

df = pd.read_csv('dataset/written_name_train_v2.csv')
train_images = "dataset/train_v2/train/"+df['FILENAME']
train_labels = df['IDENTITY']


# dft = pd.read_csv('dataset/written_name_test_v2.csv')
# test_images = "dataset/test_v2/test/"+dft['FILENAME']
# test_labels = dft['IDENTITY']

dfv = pd.read_csv('dataset/written_name_validation_v2.csv')
val_images = "dataset/validation_v2/validation/"+dfv['FILENAME']
val_labels = dfv['IDENTITY']


# preprocess
def load_image(image_path, target_size=(64, 64)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize
    return img_array

def process_csv(image_path, label):
    images_normalized = []
    for i in image_path:
        
        img_array = np.array(load_image(i))
        images_normalized.append(img_array)
  
    
    label = [str(label) for label in label]
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(label)
    encoded_labels = tokenizer.texts_to_sequences(label)
   
    df = pd.DataFrame(list(zip(images_normalized, encoded_labels)), columns=['images', 'labels']) 
    return df
        
train= process_csv(train_images, train_labels)
train.to_csv('train_process_data.csv')
val= process_csv(val_images, val_labels)
val.to_csv('val_process_data.csv')


# train_images_normalized=[]
# val_images_normalized=[]
# for i in train_images:
#     train_images_normalize = np.array(load_image(i))
#     train_images_normalized.append(train_images_normalize)

# for j in val_images:
#     val_images_normalize = np.array(load_image(j))
#     val_images_normalized.append(val_images_normalize)



# train_labels = [str(label) for label in train_labels]
# tokenizer = Tokenizer(char_level=True)
# tokenizer.fit_on_texts(train_labels)
# train_encoded_labels = tokenizer.texts_to_sequences(train_labels)

# val_labels = [str(label) for label in val_labels]
# tokenizer = Tokenizer(char_level=True)
# tokenizer.fit_on_texts(val_labels)
# val_encoded_labels = tokenizer.texts_to_sequences(val_labels)


# t_df = pd.DataFrame(list(zip(train_images_normalized, train_encoded_labels)), columns=['images', 'labels']) 
# t_df.to_csv('train_processdata.csv')

# v_df = pd.DataFrame(list(zip(val_images_normalized, val_encoded_labels)), columns=['images', 'labels']) 

# v_df.to_csv('valid_processdata.csv')