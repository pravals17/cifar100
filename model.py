import tensorflow as tf

def architecture_1(filters, strides, activation,  regularizer):
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 32, 32, 3], name = 'input_placeholder')
    x = x/255.0
    with tf.name_scope('architecture_1') as scope:
        hidden_1 = tf.layers.conv2d(x, filters[0], 5, strides = strides, padding='same', activation = activation, 
                                    kernel_regularizer = regularizer, name='hidden_1')
        pool_1 = tf.layers.max_pooling2d(hidden_1, 2, 2, padding='same')
        hidden_2 = tf.layers.conv2d(pool_1, filters[1], 3, strides = strides, padding='same', activation = activation, 
                                    kernel_regularizer = regularizer, name = 'hidden_2')
        pool_2 = tf.layers.max_pooling2d(hidden_2, 2,2, padding='same')
        flat = tf.layers.flatten(pool_2)
        dense_1 = tf.layers.dense(flat, 400, activation = activation)
        output = tf.layers.dense(dense_1, 100, name = 'output')
    tf.identity(output, name='output')
    y = tf.placeholder(tf.int32, [None, 100], name = 'output_placeholder')
    return x,y,output

def architecture_2(filters, strides, activation):
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 32, 32, 3], name = 'input_placeholder')
    x = x/255.0
    with tf.name_scope('architecture_2') as scope:
        hidden_1 = tf.layers.conv2d(x, filters[0], 5, strides = strides, padding='same', 
                                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.1), name='hidden_1')
        hidden_1 = tf.layers.batch_normalization(hidden_1, training = True)
        hidden_1 = activation(hidden_1)
        dropout_1 = tf.nn.dropout(hidden_1, 0.5, name='dropout_1')
        
        hidden_2 = tf.layers.conv2d(dropout_1, filters[1], 5, strides = strides, padding='same', 
                                    kernel_regularizer = tf.contrib.layers.l1_regularizer(scale=0.1), name='hidden_2')
        hidden_2 = tf.layers.batch_normalization(hidden_2, training = True)
        hidden_2 = activation(hidden_2)
        dropout_2 = tf.nn.dropout(hidden_2, 0.5, name='dropout_2')
        
        hidden_3 = tf.layers.conv2d(dropout_2, filters[2], 5, strides = 2, padding='same', 
                                    kernel_regularizer = tf.contrib.layers.l1_regularizer(scale=0.1), name='hidden_3')
        hidden_3 = tf.layers.batch_normalization(hidden_3, training = True)
        hidden_3 = activation(hidden_3)
        dropout_3 = tf.nn.dropout(hidden_3, 0.5, name='dropout_3')
        
        hidden_4 = tf.layers.conv2d(dropout_3, filters[3], 5, strides = 2, padding='same', 
                                    kernel_regularizer = tf.contrib.layers.l1_regularizer(scale=0.1), name='hidden_4')
        hidden_4 = tf.layers.batch_normalization(hidden_4, training = True)
        hidden_4 = activation(hidden_4)
        dropout_4 = tf.nn.dropout(hidden_4, 0.5, name='dropout_4')
              
        flat = tf.layers.flatten(dropout_4)
        dense_1 = tf.layers.dense(flat, 400, activation = tf.nn.elu,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                  bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1), name = 'dense_layer_1')
        dropout_dense_1 = tf.layers.dropout(dense_1, rate = 0.5, name='dropout_1')
        dense_2 = tf.layers.dense(dropout_dense_1, 200, activation = tf.nn.elu,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                  bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1), name = 'dense_layer_2')

        output = tf.layers.dense(dense_2, 100, name = 'output')
    tf.identity(output, name='output')
    y = tf.placeholder(tf.float32, [None, 100], name = 'output_placeholder')
    return x, y, output