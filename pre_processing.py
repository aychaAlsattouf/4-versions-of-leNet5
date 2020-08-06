#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image
import matplotlib.image as mpimg
import os
import csv
import zipfile
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from settings import *
import mpmath
import scipy
from PIL import Image
from scipy import ndimage


# In[2]:


#local_zip = 'E:\\NNProject\\GTSRB_Final_Training_Images.zip'
#zip_ref   = zipfile.ZipFile(local_zip, 'r')
#zip_ref.extractall('E:\\NNProject\\GTSRB_Final_Training_Images')
#zip_ref.close()


# In[ ]:


def rename_imgs(dir_path):
     if os.path.isdir(dir_path):
            m=0
            n=0
            for i, filename in enumerate(os.listdir(dirpath)):

                img_path=os.path.join(dirpath,filename)
                if img_path.endswith(".ppm"):
                    img = Image.open(img_path)
                    #print(img.size)
                    img = img.resize((32,32))
                    #print(img.size)
                    img.save(img_path)
                    #print(dirpath + "\\" + filename)
                    #print( dirpath + "\\" + str(n)+"_" +str(m)+ ".ppm")
                    os.rename(dirpath + "\\" + filename, dirpath + "\\" + str(n)+"_" +str(m)+ ".ppm")
                    m+=1
                    if (m>64) :
                        m=0
                        n+=1


# In[ ]:


def augmented_imgs(imgs_dir,imgs_number):
    
    #number of sub element
    imgs=imgs_number//30
    #imgs=imgs_number
   # print(imgs)
    datagen = ImageDataGenerator( 
        rotation_range = 40, 
        shear_range = 0.2, 
        zoom_range = 0.2, 
        horizontal_flip = True, 
        vertical_flip=True,
        brightness_range = (0.5, 1.5))
    for i in range(imgs):
        for j in range (30):
        # Loading a sample image 
            if (i<10 and j<10):
                # print("1    i: "+str(i)+"   j:"+str(j))
                #print(os.path.join(imgs_path,"0000"+str(i)+"_0000"+str(j)+".ppm"))
                img = load_img(os.path.join(imgs_path,"0000"+str(i)+"_0000"+str(j)+".ppm"))
            elif (i<10 and j>=10):
                # print("2    i: "+str(i)+"   j:"+str(j))
                #print(os.path.join(imgs_path,"0000"+str(i)+"_000"+str(j)+".ppm"))
                img = load_img(os.path.join(imgs_path,"0000"+str(i)+"_000"+str(j)+".ppm"))
            elif (i>=10 and j>=10 ):
                #print("3    i: "+str(i)+"   j:"+str(j))
                #print(os.path.join(imgs_path,"000"+str(i)+"_000"+str(j)+".ppm"))
                img = load_img(os.path.join(imgs_path,"000"+str(i)+"_000"+str(j)+".ppm"))

            elif(i>=10 and j<10):
                #print("4    i: "+str(i)+"   j:"+str(j))
                # print(os.path.join(imgs_path,"000"+str(i)+"_0000"+str(j)+".ppm"))
                img = load_img(os.path.join(imgs_path,"000"+str(i)+"_0000"+str(j)+".ppm"))
                # Converting the input sample image to an array 
            x = img_to_array(img) 
            # Reshaping the input image 
            x = x.reshape((1, ) + x.shape)
            n = 0
            for batch in datagen.flow(x, batch_size = 1, 
                                    save_to_dir =imgs_path,  
                                    save_prefix ='img', save_format ='ppm'): 
                n += 1
                # print(n)
                if n > 4: 
                    break


# In[19]:


def resize_imgs(dir_path):
    
    #resize all images to 32x32 and rename them to 100 image in batch
    #print(os.path.isdir(dirpath))
    if os.path.isdir(dirpath):
        for i, filename in enumerate(os.listdir(dirpath)):
            img_path=os.path.join(dirpath,filename)
            if img_path.endswith(".ppm"):
                img = Image.open(img_path)
                #print(img.size)
                img = img.resize((32,32))
                #print(img.size)
                img.save(img_path)


                


# In[ ]:


images_path="E:\\NNProject\\GTSRB_Final_Training_Images_new2"

augmented_folders=[]

for i in range(33,43):
    if(i<10):
        imgs_path=os.path.join(images_path,str("0000"+str(i)))
    else:
        imgs_path=os.path.join(images_path,str("000"+str(i)))
        
   
    imgs_number=  os.listdir(imgs_path)
    if(len(imgs_number)<1000):
        augmented_folders.append(imgs_path[-5:])
        print(len(imgs_number))
        augmented_imgs(imgs_path,len(imgs_number)-1)
    
print(augmented_folders)


# In[ ]:


for dirname in os.listdir(images_path):
    dirpath=os.path.join(images_path,dirname)
    resize_imgs(dirpath)
    rename_imgs(dirpath)


# In[ ]:


#prepare csv file for all folders
 #inside each folder
csv_path=images_path
with open(os.path.join(csv_path,'GT.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Filename", "ClassId"])
    for dirname in os.listdir(images_path):
        dirpath=os.path.join(images_path,dirname)
        m=0
        n=0
        if (os.path.isdir(dirpath)):
            for i, filename in enumerate(os.listdir(dirpath)):
                 if filename.endswith(".ppm"):
                        
                       # print(int(dirname[3:]))
                        writer.writerow([dirname+"\\"+str(n)+"_" +str(m)+ ".ppm", int(dirname[3:])])
                  #  print(str(n)+"_" +str(m)+ ".ppm", int(dirname[3:]))
                        m+=1

                        if (m>64) :
                            m=0
                            n+=1

