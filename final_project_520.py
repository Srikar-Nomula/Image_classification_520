#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import random
from itertools import chain


# Digit data loading 
# 

# In[8]:


training_data = pd.read_csv("C:\\Users\\Srikar Reddy\\Desktop\\Project\\digitdata\\trainingimages",skip_blank_lines=False, header=None,squeeze  = True)
training_labels  = pd.read_csv("C:\\Users\\Srikar Reddy\\Desktop\\Project\\digitdata\\traininglabels",skip_blank_lines=False, header=None)
test_images = pd.read_csv("C:\\Users\\Srikar Reddy\\Desktop\\Project\\digitdata\\testimages",skip_blank_lines=False, header=None,squeeze  = True)
test_labels = pd.read_csv("C:\\Users\\Srikar Reddy\\Desktop\\Project\\digitdata\\testlabels",skip_blank_lines=False, header=None)


# In[9]:


print(training_data[0:28], test_images[0:28])


# In[ ]:





# Each image in digit data is described as 28*28 pixel image. Which can be mapped 

# In[10]:


def transform_data(data):
    data_temp = data.copy()
    for i in range(data_temp.shape[0]):
        data_temp[i] = data_temp[i].replace(' ', '0').replace('#', '1').replace('+', '1')
    data_temp = data_temp.apply(lambda x: pd.Series(list(x)))
    data_temp = data_temp.apply(pd.to_numeric)
    return data_temp

train_digit_data= transform_data(training_data)
test_digit_data=transform_data(test_images)


# In[12]:


# Selecting 4x4 pixel blocks and their number of color pixels(1) as features
def colorpixelcount(df, image_height, feature_size):
    count_list=[]
   
    for i in range(int(image_height/feature_size)):
        for j in range(int(df.shape[1]/feature_size)):
            count=0
            for k in range((i*feature_size),(i*feature_size)+feature_size):
                for l in range((j*feature_size),(j*feature_size)+feature_size):
                    if df.iloc[k,l]==1:
                        count=count+1
            count_list.append(count)
  
    return count_list

def perceptron_features(data_temp, image_height, feature_size):

    feat_list=[]
    re = 0
    while re<int(len(data_temp)/image_height):  
        
        temp=colorpixelcount(data_temp.iloc[(image_height*re):(image_height*(re+1)),:].reset_index(drop=True), image_height, feature_size)
        feat_list.append(temp)
        re=re+1
    return feat_list


# In[13]:


def perceptron_train(data_for_train, classes, iterations, labels,image_height, feature_size):
    
    start_time = time.time()
    perc_features = np.asarray(perceptron_features(data_for_train, image_height, feature_size))
    weights = np.ones((classes,perc_features.shape[1]))
    trained_vector = np.zeros((classes,1))
 
    for _ in range(iterations):
        for image_no in range(len(labels)):
            temp_output = np.dot(perc_features[image_no], weights.T)
            
            # get the highest predicted class value from the predicted 
            class_temp = np.where(max(temp_output) == temp_output)[0][0]
            if class_temp != labels[image_no]:
                weights[class_temp] -= perc_features[image_no]
                weights[labels[image_no]] += perc_features[image_no]
                
    end_time = time.time()
    return weights, end_time-start_time
            
       


# In[14]:


def perceptron_test(test_digit_data,weights, test_labels, image_height, feature_size):
    
    perc_test_features = np.asarray(perceptron_features(test_digit_data,image_height, feature_size))
    labels = np.asarray(test_labels)

    misclassified = 0
    for image_no in range(len(test_labels)):
        predict_array = np.dot(perc_test_features[image_no],weights.T)

        class_temp = np.where(max(predict_array) == predict_array)[0][0]

        if class_temp != labels[image_no][0]:
            misclassified += 1
    return (1 - misclassified/len(labels))*100
        
        
# accuracy = 1 - perceptron_test(weights,test_labels)    
# accuracy


# In[176]:


def perceptron_digit(classes,image_height, feature_size):
    ratio = np.arange(0.1,1.1,0.1)
    accuracies = []
    std_accuracies = []
    times = []
    
    for value in tqdm(ratio):

        times_inner = []
        accuracies_inner =[]
        
        for _ in range(4):
            
            num_samples  =  training_labels.shape[0]
            data_random_samples = random.sample(range(num_samples), int(value * num_samples))
            data_sample_range = [ range(i * image_height, (i+1)*image_height)  for i in data_random_samples]
            data_sample_range = list(chain(*data_sample_range))
                        
            data_for_train = train_digit_data.iloc[data_sample_range]
            labels_for_train = (np.asarray(training_labels)[data_random_samples])

            weights, time = perceptron_train(data_for_train,classes,3,labels_for_train, image_height, feature_size) 
            times_inner.append(time)
            accuracies_inner.append(perceptron_test(test_digit_data,weights, test_labels, image_height, feature_size))
            
        accuracies.append(round(np.mean(accuracies_inner),3))
        std_accuracies.append(round(np.std(accuracies_inner),3))
        times.append(round(np.mean(times_inner),3))
    
    training_split = np.array([ 10 * i for i in range(1,11,1)])
    final_results = pd.DataFrame(list(zip(training_split,accuracies, std_accuracies, times)),
                                 columns = ['Training_split','Mean(Accuracy) (%)', 'Std(Accuracy) (%)','Training Time (sec)'])
   
    
    plt.plot(ratio,accuracies )
    plt.title("Training split vs Accuracy ")
    plt.xlabel("Training Split")
    plt.ylabel("Accuracy")
    plt.show()
    plt.plot(ratio,times )
    plt.title("Training time vs Training split")
    plt.xlabel("Training split")
    plt.ylabel("Time for training")
    plt.show()
    
    return final_results
perceptron_digit(10,28,4)


# Face classification for Perceptron

# In[15]:


training_face_data = pd.read_csv("C:\\Users\\Srikar Reddy\\Desktop\\Project\\facedata\\facedatatrain",skip_blank_lines=False, header=None,squeeze  = True)
training_face_labels  = pd.read_csv("C:\\Users\\Srikar Reddy\\Desktop\\Project\\facedata\\facedatatrainlabels",skip_blank_lines=False, header=None)
test_face_images = pd.read_csv("C:\\Users\\Srikar Reddy\\Desktop\\Project\\facedata\\facedatatest",skip_blank_lines=False, header=None,squeeze  = True)
test_face_labels = pd.read_csv("C:\\Users\\Srikar Reddy\\Desktop\\Project\\facedata\\facedatatestlabels",skip_blank_lines=False, header=None)
# validation_face_images = pd.read_csv("C:\\Users\\Srikar Reddy\\Desktop\\Project\\facedata\\facedatavalidation",skip_blank_lines=False, header=None,squeeze  = True)
# validation_face_labels = pd.read_csv("C:\\Users\\Srikar Reddy\\Desktop\\Project\\facedata\\facedatavalidationlabels",skip_blank_lines=False, header=None)



face_data_train = transform_data(training_face_data)
face_data_test = transform_data(test_face_images)


# In[16]:


print(face_data_train.shape)
print(training_face_labels.shape)


# In[17]:


def perceptron_face(classes,image_size, feature_size):
    
    accuracies = []
    std_accuracies = []
    times = []
    ratio = np.arange(0.1,1.05,0.1)
    
    for value in tqdm(ratio):
    
        times_inner = []
        accuracies_inner =[]
        
        for _ in range(4):
            
            num_samples  =  training_face_labels.shape[0]
            data_random_samples = random.sample(range(num_samples), int(value * num_samples))
            data_sample_range = [ range(i * image_size, (i+1)*image_size)  for i in data_random_samples]
            data_sample_range = list(chain(*data_sample_range))
                        
            data_face_train = face_data_train.iloc[data_sample_range]
            labels_for_train = (np.asarray(training_face_labels)[data_random_samples])

            weights, time = perceptron_train(data_face_train,classes,1,labels_for_train, image_size, feature_size) 
            times_inner.append(time)
            accuracies_inner.append(perceptron_test(face_data_test,weights, test_face_labels, image_size, feature_size))
            
        accuracies.append(round(np.mean(accuracies_inner),3))
        std_accuracies.append(round(np.std(np.array(accuracies_inner)),3))
        times.append(round(np.mean(times_inner),3))
        
    training_split = np.array([ 10 *i for i in range(1,11,1)])
    final_results = pd.DataFrame(list(zip(training_split, accuracies, std_accuracies, times)),
                  columns = ['Training_ratio','Mean(Accuracy) %', 'Std(Accuracy) %', 'Training Time (sec)'])
   
        
    plt.plot(ratio,accuracies )
    plt.title("Training split vs Accuracy ")
    plt.xlabel("Training Split")
    plt.ylabel("Accuracy")
    plt.show()
    plt.plot(ratio,times )
    plt.title("Training time vs Training split")
    plt.xlabel("Training split")
    plt.ylabel("Time for training")
    plt.show()
    
    return final_results
        
#
perceptron_face(2,70,2)


# In[ ]:





# In[18]:


training_data = pd.read_csv("C:\\Users\\Srikar Reddy\\Desktop\\Project\\digitdata\\trainingimages",skip_blank_lines=False, header=None,squeeze  = True)
training_labels  = pd.read_csv("C:\\Users\\Srikar Reddy\\Desktop\\Project\\digitdata\\traininglabels",skip_blank_lines=False, header=None)
test_images = pd.read_csv("C:\\Users\\Srikar Reddy\\Desktop\\Project\\digitdata\\testimages",skip_blank_lines=False, header=None,squeeze  = True)
test_labels = pd.read_csv("C:\\Users\\Srikar Reddy\\Desktop\\Project\\digitdata\\testlabels",skip_blank_lines=False, header=None)
# validation_images = pd.read_csv("C:\\Users\\Srikar Reddy\\Desktop\\Project\\digitdata\\validationimages",skip_blank_lines=False, header=None,squeeze  = True)
# validation_labels = pd.read_csv("C:\\Users\\Srikar Reddy\\Desktop\\Project\\digitdata\\validationlabels",skip_blank_lines=False, header=None)


# In[19]:


train_digit_data= transform_data(training_data)
test_digit_data=transform_data(test_images)


# In[20]:


train_digit_data.shape


# We are not extracting the features separately here, which would mean each pixel is considered as an individual feature.
# 

# In[21]:


def knn_digit(train_data,k,image_height):

    train_shape = train_data.shape
    test_shape = test_digit_data.shape
    num_images = training_labels.shape[0]
    testing_num_images = test_labels.shape[0]

    # converting the testdata into numpy array and flattened for calculations
    
    test_dig_np = np.zeros(shape = (testing_num_images, test_shape[1]*image_height))

    for i in range(testing_num_images):
        test_dig_np[i] = test_digit_data.iloc[i*image_height:(i+1)*image_height,:].to_numpy().flatten()

    accuracy = []
    accuracy_std = []
    time_list = []
    
    ratio = np.arange(0.1,1.05,0.1)
    
    # flatten the training dataset
    for split in tqdm(ratio):
        accuracy_inner = []
        time_inner = []
        for _ in range(2):
            train_dig_np = np.zeros(shape=(int((train_shape[0]/image_height)*split),train_shape[1]*image_height))
            start_time = time.time()
            
            num_samples  =  training_labels.shape[0]
            data_random_samples = random.sample(range(num_samples), int(split * num_samples))
            data_sample_range = [ range(i * image_height, (i+1)*image_height)  for i in data_random_samples]
            data_sample_range = list(chain(*data_sample_range))                       
            train_temp_data = train_data.iloc[data_sample_range,:]      
            for i in range(int(num_images*split)):                                
                train_dig_np[i] = train_temp_data[i*image_height:(i+1)*image_height].to_numpy().flatten()

            error = 0
            for image_no in range(testing_num_images):

                each_image = test_digit_data[image_no*image_height:(image_no+1)*image_height]
                each_image = each_image.to_numpy().flatten()
                neighbours_vector = np.zeros(int(num_images*split))
                
                # check each test image vector to each image in  training dataset
                for  j in range(int(num_images*split)):

                    neighbours_vector[j] = ((each_image - train_dig_np[j])**2).sum()

                neighbour_df = pd.DataFrame(neighbours_vector)
                neighbour_df =  neighbour_df[0].sort_values()[0:k]

                class_labels = []
                
                for class_index in neighbour_df.index:
                    class_labels.append(training_labels.iloc[data_random_samples[class_index],0])

                predicted_class = pd.DataFrame(class_labels)
                predicted_value = predicted_class[0].value_counts().idxmax()

                # if the predicted class is not equal return training error
                if predicted_value != test_labels.iloc[image_no][0]:
                    error += 1
#                     print(predicted_value, test_labels.iloc[image_no][0])
#             print(error, testing_num_images)
            end_time = time.time()
            accuracy_inner.append(1 - error/testing_num_images)
            time_inner.append(end_time - start_time)
        
        accuracy.append(np.mean(accuracy_inner)*100)  
        accuracy_std.append(np.std(accuracy_inner)*100)
        time_list.append(np.mean(time_inner))
       
    training_split = [ 10 * i for i in range(1,11,1)]   
    final_results = pd.DataFrame(list(zip(training_split,accuracy, accuracy_std, time_list)),
                  columns = ['Training_split','Mean(Accuracy) %', 'Standard Deviation(Accuracy) %',
                             'Training Time (sec)'])
    
        
    plt.plot(ratio,accuracy )
    plt.title("Training split vs Accuracy ")
    plt.xlabel("Training Split")
    plt.ylabel("Accuracy")
    plt.show()
    plt.plot(ratio,time_list )
    plt.title("Training time vs Training split")
    plt.xlabel("Training split")
    plt.ylabel("Time for training")
    plt.show()
    
    return final_results

knn_digit(train_digit_data,10,28)


# In[ ]:








# In[111]:


training_face_data = pd.read_csv("C:\\Users\\Srikar Reddy\\Desktop\\Project\\facedata\\facedatatrain",skip_blank_lines=False, header=None,squeeze  = True)
training_face_labels  = pd.read_csv("C:\\Users\\Srikar Reddy\\Desktop\\Project\\facedata\\facedatatrainlabels",skip_blank_lines=False, header=None)
test_face_images = pd.read_csv("C:\\Users\\Srikar Reddy\\Desktop\\Project\\facedata\\facedatatest",skip_blank_lines=False, header=None,squeeze  = True)
test_face_labels = pd.read_csv("C:\\Users\\Srikar Reddy\\Desktop\\Project\\facedata\\facedatatestlabels",skip_blank_lines=False, header=None)
# validation_face_images = pd.read_csv("C:\\Users\\Srikar Reddy\\Desktop\\Project\\facedata\\facedatavalidation",skip_blank_lines=False, header=None,squeeze  = True)
# validation_face_labels = pd.read_csv("C:\\Users\\Srikar Reddy\\Desktop\\Project\\facedata\\facedatavalidationlabels",skip_blank_lines=False, header=None)


face_data_train = transform_data(training_face_data)
face_data_test = transform_data(test_face_images)


# In[163]:


print(training_face_labels.shape)


# In[185]:


def knn_face(train_data,k,image_height):
    train_shape = train_data.shape
    test_shape = face_data_test.shape
    num_images = training_face_labels.shape[0]
    testing_num_images = test_face_labels.shape[0]

    # converting the testdata into numpy array and flattened for calculations
#     train_face_np = np.zeros(shape=(int(train_shape[0]/image_height),train_shape[1]*image_height))
#     test_face_np = np.zeros(shape = (testing_num_images, test_shape[1]*image_height))    
    test_face_np = np.zeros(shape = (testing_num_images, 900))         
    for i in range(testing_num_images):
        test_face_np[i] = face_data_test.iloc[i*image_height +25 : i*image_height + 55 ,15:45].to_numpy().flatten()
    accuracy = []
    accuracy_std = []
    time_list = []  
    ratio = np.arange(0.1,1.05,0.1)   
    
    # flatten the training dataset
    for split in ratio:
        accuracy_inner = []
        time_inner = []
        for _ in range(5):         
            start_time = time.time()
            train_face_np = np.zeros(shape=(int((train_shape[0]/image_height)*split),900))    
            data_random_samples = random.sample(range(num_images), int(split * num_images))
            data_sample_range = [ range(i * image_height, (i+1)*image_height)  for i in data_random_samples]
            data_sample_range = list(chain(*data_sample_range))
            train_face_data = train_data.iloc[data_sample_range,:]       
            for i in range(int(num_images*split)):
                train_face_np[i] = train_face_data.iloc[i*image_height +25 : i*image_height +55, 15:45].to_numpy().flatten()

            error = 0
            for image_no in range(testing_num_images):
                each_image = test_face_np[image_no]
                neighbours_vector = np.zeros(int(num_images*split))
                for  j in range(int(num_images*split)):
                    neighbours_vector[j] = ((each_image - train_face_np[j])**2).sum()
                neighbour_df = pd.DataFrame(neighbours_vector)
                neighbour_df =  neighbour_df[0].sort_values()[0:k]
#                 print(neighbour_df)                
                class_labels = []
                for class_index in neighbour_df.index:
                    
                    # convert index to unrandomised for the label
#                     print(data_random_samples)
#                     print(data_random_samples[class_index], "class_index")
#                     print(class_index, training_face_labels.iloc[data_random_samples[class_index],:])
                    class_labels.append(training_face_labels.iloc[data_random_samples[class_index],0])
                predicted_class = pd.DataFrame(class_labels)
                predicted_value = predicted_class[0].value_counts().idxmax()
                # if the predicted class is not equal return training error
                if predicted_value != test_face_labels.iloc[image_no,:][0]:
                    error += 1
            accuracy_inner.append((1 - error/testing_num_images)*100)
            end_time = time.time()
            time_inner.append(end_time-start_time)
        
      
        accuracy.append(round(np.mean(accuracy_inner),3))
        accuracy_std.append(round(np.std(accuracy_inner),3))
        time_list.append(np.mean(time_inner))
        if split ==1 :
            break
    
    training_split = [ 10 * i for i in range(1,11,1)]
    final_results = pd.DataFrame(list(zip(training_split,accuracy, accuracy_std, time_list)),
                  columns = ['Training_split','Mean(Accuracy) (%)', 'Std(Accuracy) (%)', 'Training Time (sec)'])
    
        
    plt.plot(ratio,accuracy )
    plt.title("Training split vs Accuracy ")
    plt.xlabel("Training Split")
    plt.ylabel("Accuracy")
    plt.show()
    plt.plot(ratio,time_list )
    plt.title("Training time vs Training split")
    plt.xlabel("Training split")
    plt.ylabel("Time for training")
    plt.show()
    
    return final_results

knn_face(face_data_train,10,70)


# In[ ]:





# In[ ]:




    

