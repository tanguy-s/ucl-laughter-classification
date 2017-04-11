
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import *
from sklearn.utils import shuffle
import csv


#hidden_size1 = 500
#hidden_size2 = 300 

hidden_size = 64


x_size = 48
y_size = 48
num_labels = 7
flatten_to = 6*6*128
#flatten_to = 3*3*256


tf.reset_default_graph()

## INPUTS ##
image_ph = tf.placeholder(tf.float32,[None, x_size, y_size,1])
label_ph = tf.placeholder(tf.float32,[None, num_labels])

W1 = tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.01))
b1 = tf.Variable(tf.constant(0.01,shape=[32]))
# Weights/Biases for 2nd conv layer
W2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.01))
b2 = tf.Variable(tf.constant(0.01,shape=[64]))

# Weights/Biases for 3rd conv layer
W3 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.01))
b3 = tf.Variable(tf.constant(0.01,shape=[128]))


W4 = tf.Variable(tf.truncated_normal([flatten_to, hidden_size], stddev=0.01))
b4 = tf.Variable(tf.constant(0.01,shape=[hidden_size]))

W5 = tf.Variable(tf.truncated_normal([hidden_size, hidden_size], stddev=0.01))
b5 = tf.Variable(tf.constant(0.01,shape=[hidden_size]))


W6 = tf.Variable(tf.truncated_normal([hidden_size, num_labels], stddev=0.01))
b6 = tf.Variable(tf.constant(0.01,shape=[num_labels]))


################################# IMAGE LAYERS  #################################
#1st Convolutional Layer
conv_layer1 = tf.nn.relu(tf.nn.conv2d(image_ph,W1,padding='SAME',strides=[1,1,1,1])+b1)
maxpool_layer1=tf.nn.max_pool(conv_layer1,[1, 2, 2, 1], padding='SAME',strides=[1,2,2,1])
#2nd Convolutional Layer
conv_layer2 = tf.nn.relu(tf.nn.conv2d(maxpool_layer1,W2,padding='SAME',strides=[1,1,1,1])+b2)
maxpool_layer2=tf.nn.max_pool(conv_layer2,[1, 2, 2, 1], padding='SAME',strides=[1,2,2,1])
#3rd Convolutional Layer
conv_layer3 = tf.nn.relu(tf.nn.conv2d(maxpool_layer2,W3,padding='SAME',strides=[1,1,1,1])+b3)
maxpool_layer3=tf.nn.max_pool(conv_layer3,[1, 2, 2, 1], padding='SAME',strides=[1,2,2,1])

# #4th Convolutional Layer
# conv_layer4 = tf.nn.relu(tf.nn.conv2d(maxpool_layer3,W4,padding='SAME',strides=[1,1,1,1])+b4)
# maxpool_layer4=tf.nn.max_pool(conv_layer4,[1, 2, 2, 1], padding='SAME',strides=[1,2,2,1])


flattened = tf.reshape(maxpool_layer3, [-1, flatten_to])


hidden_layer1 = tf.nn.relu(tf.matmul(flattened, W4) + b4)


hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer1, W5) + b5)

#dropout=0.5
#flattened_drop = tf.nn.dropout(flattened, 0.5)

output_layer = tf.matmul(hidden_layer2, W6) + b6




# Loss & optimization
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=label_ph))
#opt = tf.train.RMSPropOptimizer(learning_rate = 10e-4,decay=0.99999,momentum=0.99)
opt = tf.train.AdamOptimizer(learning_rate = 0.001) #tune
train_op = opt.minimize(loss)
# Make Prediction
predict = (tf.argmax(output_layer,1))


correct_prediction = tf.equal(tf.argmax(output_layer,1), tf.argmax(label_ph,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



# Load Data
X, Y = get_image_data()
# convert to num_images,48,48,1 for tensorflow
X = X.transpose((0, 2, 3, 1))
#print ("X.shape:", X.shape)



# Shuffle Data
X, Y = shuffle(X, Y)
X = X.astype(np.float32)
#Transform Y(100, to 100,7 , num labels=7)
Y = one_hot_label(Y).astype(np.float32)

print(X.shape,Y.shape)


# Make validation set (keeping last 1263 images/labels)
X_val, Y_val = X[-5263:], Y[-5263:]

# Subtracting validation from our training set
X, Y = X[:-5263], Y[:-5263]


# File to save the model
model_dir = "trained_fe/"
model_file = model_dir + "fe_model.ckpt"



N = X.shape[0]
batch_size = 128

num_batches = int(N / batch_size)



num_epochs = 20



init = tf.global_variables_initializer()

#append_log = ("LR | Epoch | Batch/Total | Loss | Accuracy")

for LR in [0.01,0.005,0.001,0.00075,0.0005,0.00025,0.0001,0.00005]:
    #losses_per_lr = []
    #acc_per_lr = []
    acc_loss_lr =[]
    with tf.Session() as sess:
        sess.run(init)
        for ep in range(num_epochs):
            print('EPOCH:',ep)
            #reshuffle training data every epoch
            X, Y = shuffle(X, Y)
            for i in range(num_batches):
                image_batch = X[i*batch_size:(i*batch_size+batch_size)]
                label_batch = Y[i*batch_size:(i*batch_size+batch_size)]

                sess.run(train_op, feed_dict={image_ph: image_batch, label_ph: label_batch})

                if i % 20 == 0:
                    l = sess.run(loss, feed_dict={image_ph: X_val, label_ph: Y_val})
                    acc = sess.run(accuracy,feed_dict ={image_ph: X_val, label_ph: Y_val})
                    
                    
                    append_log=np.array([LR,ep,i,num_batches,l,acc])
                    #acc_loss_lr.append(append_log)
                    

                    with open('opt.csv', 'a') as myfile:
                        wr = csv.writer(myfile, lineterminator='\n')
                        wr.writerow(append_log)
                    #pred = sess.run(predict, feed_dict={image_ph: X_val, label_ph: Y_val})

                    print ("epoch:", ep, "batch:", i, "/", num_batches, "loss:", l, "accuracy:", acc)
        
        #saveModel(sess, model_file)



