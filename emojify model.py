
# coding: utf-8

# In[1]:


import keras
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import matplotlib.pyplot as plt


# In[2]:


model=Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(48,48,3)))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu',input_shape=(48,48,3)))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Flatten())

model.add(Dense(output_dim=32,activation='relu'))
model.add(Dense(output_dim=7,activation='softmax'))


model.compile(loss ='categorical_crossentropy', optimizer='adam',metrics =['accuracy'])


# In[3]:


model.summary()


# In[24]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'C:/Users/Avinash/Downloads/facial/fer2013/Training',
        target_size=(48, 48),
        batch_size=32)

validation_generator = test_datagen.flow_from_directory(
        'C:/Users/Avinash/Downloads/facial/fer2013/validation',
        target_size=(48, 48),
        batch_size=32)


# In[10]:


model.fit_generator(
        train_generator,
        steps_per_epoch=1500,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=800)


# In[11]:


#fname="model-emoji-r3-4-cnn.h5"
#model.save_weights(fname)


# In[4]:


fname="model-emoji-r3-4-cnn.h5"
model.load_weights(fname)


# In[5]:


cat=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']


# In[6]:


img=cv2.imread("F:/New folder (3)/New folder (2)/ww5.jpg")
img2=cv2.imread("F:/New folder (3)/New folder (2)/Untitled.png")
img3=cv2.imread("F:/New folder (3)/New folder (2)/Untitled2.png")
img4=cv2.imread("C:/Users/Avinash/Downloads/facial/fer2013/PublicTest/Sad/00026199.jpg",1)
img5=cv2.cvtColor(img4,cv2.COLOR_BGR2RGB)
new_array = cv2.resize(img5, (48, 48))  
new_array=new_array.reshape(-1, 48, 48, 3)
new_array.shape 


# In[7]:


plt.imshow(img5)


# In[10]:


pred=model.predict_classes(new_array)

pred[0]

cat[pred[0]]


# In[11]:


import os
from tqdm import tqdm
import cv2


# In[20]:


datadir='C:/Users/Avinash/Downloads/facial/emoji'
path=os.path.join(datadir)
emoji=[]
i=0
for img in tqdm(os.listdir(path)):
    img_array=cv2.imread(os.path.join(path,img),1)
    img5=cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB)
    img6=np.array(img5)
    new_array = cv2.resize(img6, (48, 48)) 
    #print(new_array.shape)
    emoji.append(new_array)
    


# In[21]:


plt.imshow(emoji[6])
plt.show()


# In[14]:


face_cascade=cv2.CascadeClassifier('C:/Users/Avinash/Downloads/frontalFace10/haarcascade_frontalface_default.xml')


# In[15]:


font=cv2.FONT_HERSHEY_SIMPLEX


# In[36]:


cap=cv2.VideoCapture(0)
roi_gray=[]
pr=0
while 1:
    ret,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    center=0
    for(x,y,w,h) in faces:
        if(w>h):
            big=w
        else:
            big=h
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
        
    k=cv2.waitKey(1) & 0xff
    
    
    temp=cv2.resize(cv2.UMat(roi_gray),(48,48))
    cv2.imwrite("Tempimg.jpg",roi_color)
    cv2.imwrite("face.jpg",temp)
    temp=cv2.imread("face.jpg")
    temp=temp/255
    pr=model.predict_classes(temp.reshape(1,48,48,3))
    my=pr
    my2=cat[my[0]]
    cv2.putText(img,my2,(20,20),font,1,(255,255,255),2)
    for (x,y,w,h) in faces:
        emoji1=cv2.resize(emoji[my[0]],(w,h))
        img[0:0+h,400:400+w]=emoji1
    cv2.imshow('img',img)
    #print(img.shape)
    
    if k==ord('0'):
        break
cap.release()
cv2.destroyAllWindows()

