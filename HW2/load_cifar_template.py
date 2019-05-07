import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#Step 1: define a function to load traing batch data from directory
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict
def load_training_batch(folder_path,batch_id):
    """
    Args:
        folder_path: the directory contains data files
        batch_id: training batch id (1,2,3,4,5)
    Return:
        features: numpy array that has shape (10000,3072)
        labels: a list that has length 10000
    """

    ###load batch using pickle###
    batch = unpickle(folder_path+'data_batch_'+str(batch_id))
    ###fetch features using the key ['data']###
    features = batch['data']
    ###fetch labels using the key ['labels']###
    labels = batch['labels']
    return features,labels

#Step 2: define a function to load testing data from directory
def load_testing_batch(folder_path):
    
    """
    Args:
        folder_path: the directory contains data files
    Return:
        features: numpy array that has shape (10000,3072)
        labels: a list that has length 10000
    """

    ###load batch using pickle###
    tbatch = unpickle(folder_path+'test_batch')

    ###fetch features using the key ['data']###
    features = tbatch['data']
    ###fetch labels using the key ['labels']###
    labels = tbatch['labels']
    return features,labels

#Step 3: define a function that returns a list that contains label names (order is matter)
"""
    airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
"""
def load_label_names():
    
#    labelnames = [airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]
    return True

#Step 4: define a function that reshapes the features to have shape (10000, 32, 32, 3)
def features_reshape(features):
    """
    Args:
        features: a numpy array with shape (10000, 3072)
    Return:
        features: a numpy array with shape (10000,32,32,3)
    """
    features = np.reshape(features,(10000,32,32,3))
    return features

##Step 5 (Optional): A function to display the stats of specific batch data.
#def display_data_stat(folder_path,batch_id,data_id):
#    """
#    Args:
#        folder_path: directory that contains data files
#        batch_id: the specific number of batch you want to explore.
#        data_id: the specific number of data example you want to visualize
#    Return:
#        None
#
#    Descrption: 
#        1)You can print out the number of images for every class. 
#        2)Visualize the image
#        3)Print out the minimum and maximum values of pixel 
#    """
#    pass
#
#Step 6: define a function that does min-max normalization on input
def normalize(x):
    """
    Args:
        x: features, a numpy array
    Return:
        x: normalized features
    """
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    return x

#Step 7: define a function that does one hot encoding on input
def one_hot_encoding(x):
    """
    Args:
        x: a list of labels
    Return:
        a numpy array that has shape (len(x), # of classes)
    """
    new_label = np.zeros((len(x),np.amax(x)+1))
    for i in range(len(x)):
        new_label[i][x]=1
    y1 = new_label
    return y1

#Step 8: define a function that perform normalization, one-hot encoding and save data using pickle
def preprocess_and_save(features,labels,filename):
    """
    Args:
        features: numpy array
        labels: a list of labels
        filename: the file you want to save the preprocessed data
    """
    dict1 = {}
    dict1['features'] = features
    dict1['labels'] = labels
    with open(filename + '.pickle', 'wb') as handle:
        pickle.dump(dict1, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Step 9:define a function that preprocesss all training batch data and test data. 
#Use 10% of your total training data as your validation set
#In the end you should have 5 preprocessed training data, 1 preprocessed validation data and 1 preprocessed test data
def preprocess_data(folder_path):
    """
    Args:
        folder_path: the directory contains your data files
    """
    
    for i in range(5):
        feature,label = load_training_batch(folder_path,i+1)
        
        feature = normalize(feature)
        feature = features_reshape(feature)
        label = one_hot_encoding(label)
        feature, X_val1, label, y_val1 = train_test_split(feature, label, test_size=0.2, random_state=1)
        if i == 0:
            feature_val = np.zeros(np.shape(X_val1))
            label_val = np.zeros(np.shape(y_val1))
        else:
            feature_val = np.concatenate((feature_val,X_val1),axis=0)
            label_val = np.concatenate((label_val,y_val1),axis=0)
        preprocess_and_save(feature,label,'train'+str(i+1))
    preprocess_and_save(feature_val,label_val,'validation')
    tfeature,tlabel = load_testing_batch(folder_path)
    tfeature = normalize(tfeature)
    tfeature = features_reshape(tfeature)
    tlabel = one_hot_encoding(tlabel)
    preprocess_and_save(tfeature,tlabel,'test')
    print('finish preprocessing')
    
    
        
#Step 10: define a function to yield mini_batch
def mini_batch(features,labels,mini_batch_size):
    """
    Args:
        features: features for one batch
        labels: labels for one batch
        mini_batch_size: the mini-batch size you want to use.
    Hint: Use "yield" to generate mini-batch features and labels
    """
    for i in range(len(features)):
        if i % mini_batch_size == 0:
            yield features[i],labels[i]
        
        
#Step 11: define a function to load preprocessed training batch, the function will call the mini_batch() function
def load_preprocessed_training_batch(batch_id,mini_batch_size):
    """
    Args:
        batch_id: the specific training batch you want to load
        mini_batch_size: the number of examples you want to process for one update
    Return:
        mini_batch(features,labels, mini_batch_size)
    """
    file_path = ''
    batch = unpickle(file_path+'train'+str(batch_id)+'.pickle')
    features = batch['features']
    labels = batch['labels']
    
    return mini_batch(features,labels,mini_batch_size)

#Step 12: load preprocessed validation batch
def load_preprocessed_validation_batch():
    file_name = ''
    batch = unpickle(file_name+'validation')
    features,labels = batch['features'],batch['labels']
    return features,labels

#Step 13: load preprocessed test batch
def load_preprocessed_test_batch(test_mini_batch_size):
    file_name = ''
    batch = unpickle(file_name+'test')
    features,labels = batch['features'],batch['labels']
    return mini_batch(features,labels,test_mini_batch_size)

