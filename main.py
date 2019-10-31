import util
import model

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import math
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import os
import sys

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse479/shared/homework/02/', 'directory where MNIST is located')
flags.DEFINE_string('save_dir', '/lustre/work/cse479/praval395/homework2_mine/', 'directory where model graph and weights are saved')
flags.DEFINE_integer('batch_size', 64, '')
flags.DEFINE_integer('max_epoch_num', 500, '')
FLAGS = flags.FLAGS
    
def get_cross_entropy(y, output):
    return(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=output, name='cross_entropy'))

def regularize(cross_entropy, reg_coeff):
    Reg_coeff = reg_coeff
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    if regularization_losses:
        net_loss = cross_entropy + Reg_coeff * sum(regularization_losses)
    else:
        net_loss = cross_entropy
    return net_loss

def regularize_autoencoder(code, output, x):
    sparsity_weight = 5e-4
    sparsity_loss = tf.norm(code, ord=1, axis=1) #l1 regularizer loss
    reconstruction_loss = tf.reduce_mean(tf.square(output - x)) # Mean Square Error
    return(reconstruction_loss + sparsity_weight * sparsity_loss)
    
    
    
def train_operation(learning_rate ):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    return optimizer

def confusion_matrix(y, output):
    return(tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=100))

def model_saver(session, global_step_tensor):
    saver = tf.train.Saver()
    saver.save(session, os.path.join(FLAGS.save_dir, "homework2_mine"), global_step=global_step_tensor)
    
def train(x, y, session, x_train, y_train, train_op, cross_entropy, output):
    batch_size = FLAGS.batch_size
    ce_train = []
    y_preds = []
    train_num_examples = x_train.shape[0]
    for i in range(math.ceil(train_num_examples / batch_size)):
        batch_xs = x_train[i * batch_size:(i+1) * batch_size, :]
        batch_ys = y_train[i*batch_size:(i+1) * batch_size, :]
        batch_xs = np.reshape(batch_xs, (-1, 32, 32, 3))
        _, ce_loss, y_predictions = session.run([train_op, cross_entropy, output], {x:batch_xs, y:batch_ys})
        ce_train.append(ce_loss)
        y_preds.append(y_predictions)
        
    avg_cv_train = np.mean(np.concatenate(ce_train).ravel())
    train_accuracy = accuracy_score(np.argmax(y_train, axis = 1), np.argmax(np.vstack(y_preds), axis = 1))
    return avg_cv_train, train_accuracy

def test(x,y, session, x_test, y_test, cross_entropy, output):
    batch_size = FLAGS.batch_size
    ce_test = []
    y_preds = []
    conf_mxs = []
    test_num_examples = x_test.shape[0]
    conf_matx = confusion_matrix(y, output)
    for i in range(math.ceil(test_num_examples / batch_size)):
        batch_xs = x_test[i*batch_size:(i+1)*batch_size,:]
        batch_ys = y_test[i*batch_size:(i+1)*batch_size,:]
        batch_xs = np.reshape(batch_xs, (-1, 32, 32, 3))
        ce_loss, y_predictions, conf_matrix = session.run([cross_entropy, output, conf_matx], {x:batch_xs, y:batch_ys})
        ce_test.append(ce_loss)
        y_preds.append(y_predictions)
        conf_mxs.append(conf_matrix)
    avg_ce_test = np.mean(np.concatenate(ce_test).ravel())
    test_accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(np.vstack(y_preds), axis=1))
    return y_preds, avg_ce_test, test_accuracy, conf_mxs
    
def main(argv):
    
    train_accuracies_list = []
    test_accuracies_list = []
    train_losses_list = []
    test_losses_list = []
    
    #fetch data
    train_x = np.load(FLAGS.data_dir + 'cifar_images.npy')
    train_y = np.load(FLAGS.data_dir + 'cifar_labels.npy')
    train_ae = np.load(FLAGS.data_dir + 'imagenet_images.npy')
    
    # one hot encode labels
    train_y = util.one_hot_encode(train_y, 100)
    
    #split into train data and test data where test data is used for validation
    x_train, x_test, y_train, y_test = util.data_split(train_x, train_y, 0.1)
    batch_size = FLAGS.batch_size
    
    tf.reset_default_graph() 
    
    #build model according to command line argument
    if argv[1] == '1':
        x,y,output = model.architecture_1([16,32], 2,  activation = tf.nn.relu, regularizer = tf.contrib.layers.l2_regularizer(0.01))
        cross_entropy = get_cross_entropy(y, output)
        total_regularization_loss = regularize(cross_entropy, reg_coeff = 0.01)
        optimizer = train_operation(learning_rate = 0.00001 )
        train_op = optimizer.minimize(total_regularization_loss)
        
    elif argv[1] == '2':
        x,y,output = model.architecture_2([64,128, 128,256, 256, 512], 2,  activation = tf.nn.relu)
        cross_entropy = get_cross_entropy(y, output)
        total_regularization_loss = regularize(cross_entropy, reg_coeff = 0.01)
        optimizer = train_operation(learning_rate = 0.0001)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_regularization_loss)
    else:
        print("Invalid argument")
    
    
    patience = 0
    best_loss = 1000
    with tf.Session() as session:
        global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
        saver = tf.train.Saver()
        session.run(tf.global_variables_initializer())      

        for epoch in range(FLAGS.max_epoch_num):
            print("EPOCH: " + str(epoch))
            ce_loss_train, train_accuracy = train(x, y, session, x_train, y_train, train_op, cross_entropy, output)
            print("Train CE Loss: " + str(ce_loss_train))
            print("Train Accuracy: " +str(train_accuracy))
            train_accuracies_list.append(train_accuracy)
            train_losses_list.append(ce_loss_train)
            y_preds, ce_loss_test, test_accuracy, conf_mxs = test(x,y,session, x_test, y_test, cross_entropy, output)
            test_accuracies_list.append(test_accuracy)
            test_losses_list.append(ce_loss_test)
            if ce_loss_test < best_loss:
                best_loss = ce_loss_test
                model_saver(session, global_step_tensor)
                patience = 0
            else:
                patience = patience + 1
                print("Patience: " + str(patience))
                if patience > 20:
                    break
                print("Test CE Loss: " + str(ce_loss_test))
                print("Test Accuracy: " +str(test_accuracy))
                
            error = 1 - test_accuracy
            conf_interval_upper = error + 1.96*math.sqrt((error*(1-error))/y_test.shape[0])
            conf_interval_lower = error - 1.96*math.sqrt((error*(1-error))/y_test.shape[0])

        print('upper_bound' + str(conf_interval_upper))
        print('lower_bound' + str(conf_interval_lower))

        # Generate Loss Plot
        plt.clf()     
        plt.figure(figsize=(10,6))
        plt.plot(train_losses_list, label = 'train loss')
        plt.plot(test_losses_list, label = 'test loss')
        plt.legend(loc = 'upper left')
        plt.title('Train and Test Loss')
        plt.grid()
        plt.savefig('loss_homework2.png', dpi=300)
        plt.show()

        #Generate Accuracy PLot    
        plt.figure(figsize=(10,6))
        plt.plot(train_accuracies_list, label = 'train accuracy')
        plt.plot(test_accuracies_list, label = 'test accuracy')
        plt.legend(loc = 'upper left')
        plt.title('Train and Test Accuracy')
        plt.grid()
        plt.savefig('accuracy_homework2.png', dpi=300)
        plt.show()
       
        
if __name__ == "__main__":
    tf.app.run()