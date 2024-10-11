
# coding: utf-8

# In[136]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import sys


# In[137]:


from tqdm import tqdm


# In[138]:


df=pd.read_csv("file:///C:/Users/Avinash/Downloads/challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013/fer2013/fer2013.csv")


# In[139]:



path='C:\\Users\\Avinash\\Downloads\\challenges-in-representation-learning-facial-expression-recognition-challenge\\fer2013'


# In[104]:


path


# In[46]:


os.system('mkdir{}'.format(path))


# In[131]:


if os.path.exists(path):
    print('yes')
else:
    print('no')


# In[140]:


label_names=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
data=np.genfromtxt('C:/Users/Avinash/Downloads/challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013/fer2013/fer2013.csv',delimiter=',',dtype=None,encoding='utf8')


# In[141]:


labels=data[1:,0].astype(np.int32)


# In[142]:


image_buffer=data[1:,1]


# In[143]:


image=np.array([np.fromstring(image,np.uint8,sep=' ')for image in image_buffer])


# In[110]:


image[1].shape


# In[144]:


usage=data[1:,2]


# In[145]:


dataset=zip(labels,image,usage)


# In[147]:


for i,d in tqdm(enumerate(dataset)):
    usage_path=os.path.join(path,d[-1])
    label_path=os.path.join(usage_path,label_names[d[0]])
    img=d[1].reshape((48,48))
    img_name='%08d.jpg'%i
    img_path=os.path.join(label_path,img_name)
    if not os.path.exists(usage_path):
        os.system('mkdir {}'.format(usage_path))
    if not os.path.exists(label_path):
        os.system('mkdir {}'.format(label_path))
    cv2.imwrite(img_path, img)
    #print ('Write {}'.format(img_path))


# In[ ]:




