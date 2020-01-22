# import python libraries needed
import os
import numpy as np
import tensorflow as tf
from glob import glob
from PIL import Image
from matplotlib import pyplot as plt

################################################################################
''' HYPERPARAMS '''
################################################################################

image_size = 256
batch_size = 20
num_epochs = 15
learning_rate = 1e-3
dropout_rate = .5 # percent of nodes to dropout [0,1]
use_pixel_norm = True

saved_epoch = num_epochs-1 # epoch model to test
training = True # --> True = training | False = holdout prediction

################################################################################

if not training: # testing specific hyperparams
    batch_size = 1
    dropout_rate = 0

if not os.path.exists('models'):
    os.makedirs('models')

# only use GPU 0
os.environ["CUDA_VISIBLE_DEVICES"]='0'

# data paths training
images_0 = glob('./extracted_sclerotic_gloms/0/*.jpg')
images_1 = glob('./extracted_sclerotic_gloms/1/*.jpg')
labels_0 = list(np.zeros(len(images_0)))
labels_1 = list(np.ones(len(images_1)))

images = images_0 + images_1
labels = labels_0 + labels_1

# data paths testing
images_0_test = glob('./extracted_sclerotic_gloms/holdout/0/*.jpg')
images_1_test = glob('./extracted_sclerotic_gloms/holdout/1/*.jpg')
labels_0_test = list(np.zeros(len(images_0)))
labels_1_test = list(np.ones(len(images_1)))

images_test = images_1_test + images_0_test
labels_test = labels_1_test + labels_0_test

def main(): # define main training testing loop

    #############################################################################
    ''' DATASET PIPELINE '''
    #############################################################################

    with tf.device('/cpu:0'):

        if training:
            ds_img = tf.data.Dataset.from_tensor_slices(images) # dataset from img paths
            ds_labels = tf.data.Dataset.from_tensor_slices(labels) # dataset from labels

            ''' map image dataset 2 options'''
            # OPTION 1 --> ### USE PYTHON OPS FOR MAP --> SLOW ### helpful if you need python specific ops
            #ds_img = ds_img.map(lambda filename: tf.py_function(_parse_function_python, [filename, True], tf.float32), num_parallel_calls=20) # map images to dataset

            # OPTION 2 --> ### USE TF OPS FOR MAP --> FASTER ###
            ds_img = ds_img.map(lambda filename: _parse_function_tf(filename, True), num_parallel_calls=20) # map images to dataset

            ds_labels = ds_labels.map(lambda label: tf.py_function(_parse_labels, [label], tf.int32), num_parallel_calls=20) # map labels to dataset
            ds = tf.data.Dataset.zip((ds_img,ds_labels)) # combine img and labels to keep them paired
            ds = ds.repeat(10) # repeat the dataset 10X per epoch (did this for the small dataset used)
            ds = ds.shuffle(5000) # shuffle with sudo random batch of X examples
            ds = ds.batch(batch_size, drop_remainder=True) # batch data
            ds = ds.prefetch(buffer_size=1) # prefetch 1 batch --> makes network much more efficient

            iterator = tf.data.Iterator.from_structure(ds.output_types, ds.output_shapes) # create iterator
            training_init_op = iterator.make_initializer(ds) # make iterator initializer op
            next_im, next_label = iterator.get_next() # define get_next call to iterator


        else:
            # setup dataset for testing
            ds_img = tf.data.Dataset.from_tensor_slices(images_test) # dataset from img paths
            ds_labels = tf.data.Dataset.from_tensor_slices(labels_test) # dataset from labels

            ds_img = ds_img.map(lambda filename: _parse_function_tf(filename, False), num_parallel_calls=20) # map images to dataset

            ds_labels = ds_labels.map(lambda label: tf.py_function(_parse_labels, [label], tf.int32), num_parallel_calls=20) # map labels to dataset
            ds = tf.data.Dataset.zip((ds_img,ds_labels)) # combine img and labels to keep them paired
            ds = ds.batch(batch_size) # batch data
            ds = ds.prefetch(buffer_size=1) # prefetch 1 batch --> makes network much more efficient

            iterator = tf.data.Iterator.from_structure(ds.output_types, ds.output_shapes) # create iterator
            training_init_op_holdout = iterator.make_initializer(ds) # make iterator initializer op
            next_im, next_label = iterator.get_next() # define get_next call to iterator


    #############################################################################
    ''' SETUP NETWORK ARCHITECTURE '''
    #############################################################################

    x_data = tf.reshape(next_im, [batch_size,image_size,image_size,3])
    # batch x 256x256x3
    print(x_data.shape)

    #########################################################
    ### Convolution block ###
    #########################################################

    # convolution = expand feature map dim
    x_data = tf.layers.conv2d(x_data, filters=16, kernel_size=5, padding='same' )

    # normalize feature values using custom op --> op defined below
    if use_pixel_norm:
        x_data = pixel_norm(x_data)

    # nonlinearity = leaky relu does not have disapearing gradient problem
    x_data = tf.nn.leaky_relu(x_data)

    # max pool = reduce 2x2 blocks to only max number = reduce x+y dims
    x_data = tf.layers.max_pooling2d(x_data, pool_size=2, strides=2,padding='same')

    # dropout
    x_data = tf.layers.dropout(x_data, rate=dropout_rate)

    # batch x 128x128x32
    print(x_data.shape)

    #########################################################
    ### repeat convolution block until desired size
    ### this could be done by defineing conv_block as a function and placing it in a for loop
    #########################################################

    x_data = tf.layers.conv2d(x_data, filters=32, kernel_size=5, padding='same' )
    if use_pixel_norm: x_data = pixel_norm(x_data)
    x_data = tf.nn.leaky_relu(x_data)
    x_data = tf.layers.max_pooling2d(x_data, pool_size=2, strides=2,padding='same')
    x_data = tf.layers.dropout(x_data, rate=dropout_rate)
    # batch x 64x6x64
    print(x_data.shape)

    #########################################################

    x_data = tf.layers.conv2d(x_data, filters=64, kernel_size=5, padding='same' )
    if use_pixel_norm: x_data = pixel_norm(x_data)
    x_data = tf.nn.leaky_relu(x_data)
    x_data = tf.layers.max_pooling2d(x_data, pool_size=2, strides=2,padding='same')
    x_data = tf.layers.dropout(x_data, rate=dropout_rate)
    # batch x 32x32x128
    print(x_data.shape)

    #########################################################

    x_data = tf.layers.conv2d(x_data, filters=128, kernel_size=5, padding='same' )
    if use_pixel_norm: x_data = pixel_norm(x_data)
    x_data = tf.nn.leaky_relu(x_data)
    x_data = tf.layers.max_pooling2d(x_data, pool_size=2, strides=2,padding='same')
    x_data = tf.layers.dropout(x_data, rate=dropout_rate)
    # batch x 16x16x256
    print(x_data.shape)

    #########################################################

    x_data = tf.layers.conv2d(x_data, filters=256, kernel_size=5, padding='same' )
    if use_pixel_norm: x_data = pixel_norm(x_data)
    x_data = tf.nn.leaky_relu(x_data)
    x_data = tf.layers.max_pooling2d(x_data, pool_size=2, strides=2,padding='same')
    x_data = tf.layers.dropout(x_data, rate=dropout_rate)
    # batch x 8x8x512
    print(x_data.shape)

    #########################################################

    x_data = tf.layers.conv2d(x_data, filters=256, kernel_size=5, padding='same' )
    if use_pixel_norm: x_data = pixel_norm(x_data)
    x_data = tf.nn.leaky_relu(x_data)
    x_data = tf.layers.max_pooling2d(x_data, pool_size=2, strides=2,padding='same')
    x_data = tf.layers.dropout(x_data, rate=dropout_rate)
    # batch x 4x4x512
    print(x_data.shape)

    #########################################################
    ### flatten data --> transform to "feature vector"
    #########################################################

    x_data = tf.reshape(x_data, [batch_size,-1])
    print(x_data.shape)

    #########################################################
    ### MLP --> classifier using dense (fully connected layers)
    #########################################################

    x_data = tf.layers.dense(x_data,512)
    x_data = tf.nn.leaky_relu(x_data)
    x_data = tf.layers.dropout(x_data, rate=dropout_rate)
    print(x_data.shape)
    out = tf.layers.dense(x_data,2)
    print(out.shape)

    #########################################################
    ### scale output with softmax for display
    #########################################################

    scaled_out = tf.nn.softmax(out)

    #########################################################
    ### DEFINE NETWORK LOSS
    #########################################################

    # cross entropy = how well we predict
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=next_label, logits=out))

    # L2 norm regularization = penalize for having large parameter values
    vars   = tf.trainable_variables()
    lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars ]) * 0.001

    cost = cross_entropy + lossL2

    #########################################################
    ### define BACKWARDS PASS
    ### tf does automatic differentiation
    #########################################################

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    #########################################################
    ### op to save and restore variables
    #########################################################

    saver = tf.train.Saver()


    if training: # training op
        #############################################################################
        ''' TRAINING LOOP '''
        #############################################################################

        init = tf.global_variables_initializer() # init all the variables in graph randomly

        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(num_epochs):
                print('starting epoch {}\n'.format(epoch))

                # init iterator --> setup iterator
                sess.run(training_init_op)

                iter = 0
                losses = []
                L2_losses = []
                while True: # loop forever untill break in except
                    iter += 1
                    try: # run training op
                        if iter % 20 == 0: # print cross_entropy every X steps
                            _, loss, L2 = sess.run([train_op, cross_entropy, lossL2])
                            losses.append(loss)
                            L2_losses.append(L2)
                            print(loss)
                        else: # just run optimizer op | dont print anything
                            _, loss, L2 = sess.run([train_op, cross_entropy, lossL2])
                            losses.append(loss)
                            L2_losses.append(L2)

                    except tf.errors.OutOfRangeError: # no more data in itererator --> end of epoch
                        saver.save(sess , '{}/saved_model_epoch_{}'.format('models', epoch))
                        print('\nDone with epoch {}\nmodel saved.\n\n###################################################\n'.format(epoch))
                        break

            # plot losses
            CE = plt.plot(losses)
            L2 = plt.plot(L2_losses)
            plt.legend([CE, L2], ['cross entropy loss', 'L2 loss'])
            plt.title('network training losses')
            plt.show()


    else: # testing
        #############################################################################
        ''' TESTING LOOP '''
        #############################################################################

        with tf.Session() as sess:
            saver.restore(sess, '{}/saved_model_epoch_{}'.format('models', saved_epoch))

            # init dataset iterator
            sess.run(training_init_op_holdout)

            preds=[]
            reals=[]

            while True:
                try:
                    classification, real_labels = sess.run([scaled_out,next_label])
                    pred = np.argmax(classification[0,:])
                    real = real_labels[0]
                    print('predicted: {} | real: {}'.format(pred, real))
                    preds.append(pred)
                    reals.append(real)

                except tf.errors.OutOfRangeError: # end of epoch
                    print('\ndone testing...\n')
                    # calculate stats
                    reals = np.array(reals)
                    preds = np.array(preds)

                    def inv(x):
                        return np.abs(x-1)

                    TP = float(np.sum(reals*preds))
                    TN = float(np.sum(inv(reals) * inv(preds)))
                    FP = float(np.sum(inv(reals) * preds))
                    FN = float(np.sum(reals * inv(preds)))

                    # sensitivity
                    TPR = TP/(TP + FN)

                    # specificity
                    TNR = TN/(TN + FP)

                    print('sensitivity: {}\nspecificity: {}'.format(TPR,TNR))

                    break


#----------------------------------------------------------------------------
# Pixelwise feature vector normalization.

'''
This is a custom op which takes the feature outputs
after convolution and normalizes them

this is done by dividing them by the sqrt of the MSE

helps to keep the gradients and feature values in check
durring prediction and training
'''

def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + epsilon)

#----------------------------------------------------------------------------
# parse image path to image using tensorflow ops

def _parse_function_tf(path, augment=False):
    # open image path
    im = tf.io.read_file(path)

    # decode image data
    im = tf.io.decode_jpeg(im)

    # resize image
    im = tf.image.resize_images(im, [image_size,image_size])

    # augment images at runtime
    if augment:
        # rotate image in 90 deg incraments
        num_rot = np.int32(round(np.random.uniform()*2))
        im = tf.image.rot90(im, k=num_rot)

        # randomly flip image
        im = tf.image.random_flip_left_right(im)
        im = tf.image.random_flip_up_down(im)

    # normalize between -1 and 1
    im = (im/127.5)-1
    return im

#----------------------------------------------------------------------------
# parse image path to image using python ops

def _parse_function_python(path, augment=False):
    path = path.numpy() # convert tenor to string

    # open image using PIL
    im = Image.open(path)

    # resize image object
    im =  im.resize((image_size,image_size), resample=0)

    # augment images at runtime
    if augment:
        # rotate image in 90 deg incraments
        deg = round(np.random.uniform()*3)*90
        im.rotate(deg)

        # randomly flip image
        if np.random.uniform() > .5:
            im.transpose(Image.FLIP_LEFT_RIGHT)

        if np.random.uniform() > .5:
            im.transpose(Image.FLIP_TOP_BOTTOM)

    # convert image object to np array
    im = np.array(im)

    # normalize between -1 and 1
    im = (im/127.5)-1
    return im

#----------------------------------------------------------------------------
# parse labels convert to int32

def _parse_labels(label):
    label = np.int32(label)
    return label

#----------------------------------------------------------------------------
# run main()

if __name__ == '__main__':
    main()
