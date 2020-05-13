import numpy as np
import pandas as pd
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

#data = pd.read_csv('list_attr.csv')
#data = data.iloc[:200,:]
#data=data.replace(to_replace = -1,value='nan')
#data=data.loc[:, :].replace(1, pd.Series(data.columns, data.columns))
data=data.set_index('image_id').T.to_dict('list')
#print(data)
#f= open("captions.txt","w+")
#with open("captions.txt", "a") as myfile:
#    myfile.write(data)
data = pd.read_csv('captions.txt')

def refining_img():
    train_images = []
    for i in tqdm(os.listdir(train_data)):
        path = os.path.join(train_data,i)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize((cv2.cvtColor(img,cv2.COLOR_BGR2RGB)),(100, 100))
       #    img = cv2.resize(img,(100,100))
            train_images.append([np.array(img)])
        else:
            print('Image not loaded')
    return train_images

training_images = np.asarray(refining_img())
training_images= np.array([i[0] for i in training_images]).reshape(-1,100,100,3)

plt.figure(figsize=(40, 40))
for i in range(10):
    # display original images
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(training_images[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)