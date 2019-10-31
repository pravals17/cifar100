from sklearn.model_selection import train_test_split
import numpy as np

def data_split(x,y, size):
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=size)
    return X_train, X_test, y_train, y_test

def one_hot_encode(data,num_classes):
    data = np.array(data,dtype=np.int8).reshape(-1)
    return np.eye(num_classes)[data]