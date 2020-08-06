#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras import optimizers
import os
import keras
from keras.models import Model
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import plot_model
import numpy as np
from keras import layers
from settings import *
from PIL import Image
from keras.layers.merge import concatenate
from keras.models import load_model


# In[2]:


keras.backend.clear_session()


# # Original leNet5

# In[ ]:


def create_original_model():
    input_data= layers.Input(shape=(32,32,3))
    conv_1=layers.Conv2D(filters=6, kernel_size=5,padding="VALID",strides=(1, 1),activation="relu")(input_data)
    P1=layers. MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid')(conv_1)
    
    conv_2=layers.Conv2D(filters=16, kernel_size=5,padding="VALID",strides=(1, 1),activation="relu")(P1)
    P2=layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid')(conv_2)
    
    F2=layers.Flatten()(P2)
    
    
    D1=layers.Dense(units=120, activation='relu')(F2)
    
    D2=layers.Dense(units=84, activation='relu')(D1)
  
    D3=layers.Dense(units=43, activation = 'softmax')(D2)
    
    model = Model(inputs=input_data, outputs=D3)
    
    return model


# In[ ]:


original_model=create_original_model()
plot_model(original_model, to_file='original_model.png', show_shapes=True, show_layer_names=True)


# # Changed architecture

# In[3]:


def create_changed_model():
    input_data= layers.Input(shape=(32,32,3))
    conv_1=layers.Conv2D(filters=6, kernel_size=5,padding="VALID",strides=(1, 1),activation="relu")(input_data)
    P1=layers. MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid')(conv_1)
    
    conv_2=layers.Conv2D(filters=16, kernel_size=5,padding="VALID",strides=(1, 1),activation="relu")(P1)
    P2=layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid')(conv_2)
    
    F2=layers.Flatten()(P2)
    F1=layers.Flatten()(P1)
    Merge=concatenate([F1, F2])
    
    
    D1=layers.Dense(units=120, activation='relu')(Merge)
    
    D2=layers.Dense(units=84, activation='relu')(D1)
  
    D3=layers.Dense(units=43, activation = 'softmax')(D2)
    
    model = Model(inputs=input_data, outputs=D3)
    
    return model


# In[4]:


changed_model=create_changed_model()
plot_model(changed_model, to_file='Changed_architecture_model.png', show_shapes=True, show_layer_names=True)


# # Dropout on fully connected layers

# In[ ]:


#define keras model
def create_dropout_fully_connected_model():
    
    input_data= layers.Input(shape=(32,32,3))
    conv_1=layers.Conv2D(filters=6, kernel_size=5,padding="VALID",strides=(1, 1),activation="relu")(input_data)
    P1=layers. MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid')(conv_1)
    
    conv_2=layers.Conv2D(filters=16, kernel_size=5,padding="VALID",strides=(1, 1),activation="relu")(P1)
    P2=layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid')(conv_2)
    
    F2=layers.Flatten()(P2)
    F1=layers.Flatten()(P1)
    Merge=concatenate([F1, F2])
    
    Merge_dropout=layers.Dropout(rate=0.1)(Merge)
    
    D1=layers.Dense(units=120, activation='relu')(Merge_dropout)
    D1_dropout= layers.Dropout(rate=0.3)(D1)
    
    D2=layers.Dense(units=84, activation='relu')(D1_dropout)
    D2_dropout=layers.Dropout(rate=0.5)(D2)
    
    D3=layers.Dense(units=43, activation = 'softmax')(D2_dropout)
    
    model = Model(inputs=input_data, outputs=D3)
        
        
    return model


# In[ ]:


dropout_fully_connected_model=create_dropout_fully_connected_model()
plot_model(dropout_fully_connected_model, to_file='Dropout_fully_connected _model.png', show_shapes=True, show_layer_names=True)


# # Dropout on fully connected and convolutional layers

# In[ ]:


def create_dropout_fully_connected_convolutional_model():
    
    input_data= layers.Input(shape=(32,32,3))
    conv_1=layers.Conv2D(filters=6, kernel_size=5,padding="VALID",strides=(1, 1),activation="relu")(input_data)
    P1=layers. MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid')(conv_1)
    P1_dropout= layers.Dropout(rate=0.3)(P1)
    
    conv_2=layers.Conv2D(filters=16, kernel_size=5,padding="VALID",strides=(1, 1),activation="relu")(P1_dropout)
    P2=layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid')(conv_2)
    P2_dropout= layers.Dropout(rate=0.5)(P2)
    
    F2=layers.Flatten()(P2_dropout)
    F1=layers.Flatten()(P1_dropout)
    Merge=concatenate([F1, F2])
    
    Merge_dropout=layers.Dropout(rate=0.1)(Merge)
    D1=layers.Dense(units=120, activation='relu')(Merge_dropout)
    D1_dropout= layers.Dropout(rate=0.3)(D1)
    
    D2=layers.Dense(units=84, activation='relu')(D1_dropout)
    D2_dropout=layers.Dropout(rate=0.5)(D2)
    
    D3=layers.Dense(units=43, activation = 'softmax')(D2_dropout)
    
    model = Model(inputs=input_data, outputs=D3)
        
        
    return model


# In[ ]:


dropout_fully_connected_convolutional_model=create_dropout_fully_connected_convolutional_model()
plot_model(dropout_fully_connected_convolutional_model, to_file='Dropout_fully_connected_convolutional_model.png', show_shapes=True, show_layer_names=True)


# # Load GTSRB dataset

# In[ ]:


X_train,Y_train,X_valid,Y_valid=load_data()


# In[ ]:


Y_train_hot=keras.utils.to_categorical(Y_train,num_classes=43)


# # Optimizer Configuration

# In[ ]:


adam=optimizers.Adam(lr=0.009, beta_1=0.9, beta_2=0.999, amsgrad=False)


# # Configures the models for training.

# In[ ]:


original_model.compile(optimizer="adam",
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
changed_model.compile(optimizer="adam",
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
dropout_fully_connected_model.compile(optimizer="adam",
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
dropout_fully_connected_convolutional_model.compile(optimizer="adam",
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# # Models Training

# ### training of Original model 

# In[ ]:


print('# Fit model on training data')
original_model_history = original_model.fit(X_train, Y_train,
                    batch_size=64,
                    epochs=100,
                    # We pass some validation for
                    # monitoring validation loss and metrics
                    # at the end of each epoch
                    validation_split = 0.2
                   )


# # save model and weights 

# In[ ]:


original_model.save("E:\\NNProject\\Models_and_weights\\original_model.h5")
print("Saved model to disk")


# ### training of Changed model 

# In[ ]:


print('# Fit model on training data')
changed_model_history = changed_model.fit(X_train, Y_train,
                    batch_size=64,
                    epochs=100,
                    # We pass some validation for
                    # monitoring validation loss and metrics
                    # at the end of each epoch
                    validation_split = 0.2
                   )


# ## save model and weights 

# In[ ]:


changed_model.save("E:\\NNProject\\Models_and_weights\\changed_model.h5")
print("Saved model to disk")


# ### training of Dropout fully connected model training

# In[ ]:



print('# Fit model on training data')
dropout_fully_connected_model_history = dropout_fully_connected_model.fit(X_train, Y_train,
                    batch_size=64,
                    epochs=100,
                    # We pass some validation for
                    # monitoring validation loss and metrics
                    # at the end of each epoch
                    validation_split = 0.2
                   )


# # save model and weights 

# In[ ]:


dropout_fully_connected_model.save("E:\\NNProject\\Models_and_weights\\dropout_fully_connected_model.h5")
print("Saved model to disk")


# ### training of Dropout_fully_connected_convolutional_model 

# In[ ]:


print('# Fit model on training data')
dropout_fully_connected_convolutional_model_history = dropout_fully_connected_convolutional_model.fit(X_train, Y_train,
                    batch_size=64,
                    epochs=100,
                    # We pass some validation for
                    # monitoring validation loss and metrics
                    # at the end of each epoch
                    validation_split = 0.2
                   )


# ## save model and weights 

# In[ ]:


dropout_fully_connected_convolutional_model.save("E:\\NNProject\\Models_and_weights\\dropout_fully_connected_convolutional_model.h5")
print("Saved model to disk")


# # evaluation of Test dataset

# In[ ]:


original_model_score = original_model.evaluate(X_valid, Y_valid, batch_size=64)


# In[ ]:


changed_model_score = changed_model.evaluate(X_valid, Y_valid, batch_size=64)


# In[ ]:


dropout_fully_connected_model_score = dropout_fully_connected_model.evaluate(X_valid, Y_valid, batch_size=64)


# In[ ]:


dropout_fully_connected_convolutional_model_score = dropout_fully_connected_convolutional_model.evaluate(X_valid, Y_valid, batch_size=64)


# # Plot

# In[ ]:


original_model_acc      = original_model_history.history[     'acc' ]
original_model_val_acc  = original_model_history.history[ 'val_acc' ]
original_model_loss     = original_model_history.history[    'loss' ]
original_model_val_loss = original_model_history.history['val_loss' ]


# In[ ]:


changed_model_acc      = changed_model_history.history[     'acc' ]
changed_model_val_acc  = changed_model_history.history[ 'val_acc' ]
changed_model_loss     = changed_model_history.history[    'loss' ]
changed_model_val_loss = changed_model_history.history['val_loss' ]


# In[ ]:


dropout_fully_connected_model_acc      = dropout_fully_connected_model_history.history[     'acc' ]
dropout_fully_connected_model_val_acc  = dropout_fully_connected_model_history.history[ 'val_acc' ]
dropout_fully_connected_model_loss     = dropout_fully_connected_model_history.history[    'loss' ]
dropout_fully_connected_model_val_loss = dropout_fully_connected_model_history.history['val_loss' ]


# In[ ]:


dropout_fully_connected_convolutional_model_acc      = dropout_fully_connected_convolutional_model_history.history[     'acc' ]
dropout_fully_connected_convolutional_model_val_acc  = dropout_fully_connected_convolutional_model_history.history[ 'val_acc' ]
dropout_fully_connected_convolutional_model_loss     = dropout_fully_connected_convolutional_model_history.history[    'loss' ]
dropout_fully_connected_convolutional_model_val_loss = dropout_fully_connected_convolutional_model_history.history['val_loss' ]


# In[ ]:


epochs   = range(len(changed_model_acc))
plt.plot  ( epochs,     original_model_val_loss ,'b' )
plt.plot  ( epochs, changed_model_val_loss,'r' )
plt.plot  ( epochs, dropout_fully_connected_model_val_loss,'g' )
plt.plot  ( epochs, dropout_fully_connected_convolutional_model_val_loss ,'y' )
plt.xlabel('epochs', fontsize=16)
plt.ylabel('loss', fontsize=16)
plt.title ('validation loss')
plt.figure()


plt.plot  ( epochs,     original_model_val_acc ,'b' )
plt.plot  ( epochs, changed_model_val_acc,'r' )
plt.plot  ( epochs, dropout_fully_connected_model_val_acc,'g' )
plt.plot  ( epochs, dropout_fully_connected_convolutional_model_val_acc ,'y' )
plt.xlabel('epochs', fontsize=16)
plt.ylabel('accuracy', fontsize=16)
plt.title ('validation accuracy'   )
plt.figure()


# # Upload model

# In[ ]:


# load model
#path to the wanted h5 file
h5_path=""
model = load_model(h5_path)


# 
# # Prediction

# In[ ]:


folder_path="E:\\NNProject\\test_data"
M= len([name for name in os.listdir(folder_path) if os.path.join(folder_path,name).endswith(".ppm")])
ncols=5
columns=1
rows=0
nrows=(M//5)+1

img_width,img_height=32,32
fig,axes = plt.subplots(nrows, ncols, figsize=(100,100))
i=0
#plt.rcParams.update({'font.size': 12})
print("columns : ",columns,"   rows:",rows,"    i:",i )
for img in os.listdir(folder_path):
    img_path = os.path.join(folder_path, img)
    img = Image.open(img_path)
    #print(img.size)
    img = img.resize((img_width,img_height))
    #print(img.size)
    img.save(img_path)
    image = mpimg.imread(img_path)
    im = np.array(image).reshape(1, img_width,img_height,3)
    
    
    y=dropout_fully_connected_model.predict(im)
    
    index=np.where(y[0]==np.amax(y))
    label=index[0][0]
    #print(label)
    name= get_name_by_index(label)
    #print("i:",i,"     rows:",rows)
    if(i >= 5):
        #print("lllllllll")
        i=0
        rows+=1
    axes[rows,i].imshow(image)
    #need to incease font size
    
    axes[rows,i].set_xlabel(name)
   #or
    #axes[rows,i].set_title(name)
    i+=1
        
    
    
    
plt.show()


# ## Visualization 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


layer_outputs = [layer.output for layer in model.layers[1:13]]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
print(type(activation_model))


# In[ ]:


img_path="E:\\NNProject\\GTSRB_Final_Test_Images\\00011.ppm"

img = Image.open(img_path)
#print(img.size)
img = img.resize((32,32))
#print(img.size)
img.save(img_path)
image = mpimg.imread(img_path)
ima = np.array(image).reshape(1, 32,32,3)


# In[ ]:


activations = activation_model.predict(ima) # Returns a list of five Numpy arrays: one array per layer activation


# In[ ]:


first_layer_activation = activations[0]
#print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0,:,:,4], cmap='viridis')


# In[ ]:


layer_names = []
for layer in model.layers[:4]:
    #if(layer.name is not 'concatenate_2'):
    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
    
images_per_row = 6

for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    #print (layer_name)
    #print(type(display_grid))
    for col in range(n_cols): 
       # print(col)
        for row in range(images_per_row):
            #scale * display_grid.shape[0] == 0
            print(layer_activation.shape)
            channel_image = layer_activation[0, :, :, (col * images_per_row + row)]
            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, # Displays the grid
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
   
    plt.title(layer_name)

    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

