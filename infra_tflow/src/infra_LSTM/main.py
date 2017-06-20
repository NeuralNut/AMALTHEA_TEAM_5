#%%
"""
Created on Mon Jun  5 16:44:02 2017

@author: emilyjensen

Main file to test on UCR and infrasound LSTM datasets.
Loads data, defines hyperparameters, calls model build, trains model, and prints
results of training

Does not use GPU
"""

"""
TODO:
	Make confusion matrix rows/cols names of classes 
	log experimental params 
	save/restore scheme
    Adaptive learning/dropout rates
    Better validation scheme
"""
#import packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from datetime import datetime
import io


# import user-defined functions
from model import Model
from functions import load_data
from functions import create_batches


tf.reset_default_graph()


now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs_Mitch"
logdir = "{}/run-{}/".format(root_logdir, now)



# Load the desired dataset

direc = str(input('Where is the dataset located? '))

# Define model save path

save_path = logdir

# Splits training set into training and validation sets
ratio = 0.8
dataset='Two_Patterns'
X_train,train_seq,X_val,val_seq,X_test,test_seq,y_train,y_val,y_test = load_data(direc,ratio,dataset)

#%%
"""Define configuration of hyperparameters"""
# Define hyperparameters also used in this file
val_increment = 10 # how many epochs between checking validation preformance
batch_size = int(X_train.shape[0]/10) # adapt batch size to size of the dataset

 

max_epochs = 10
dropout = 0.8  
num_classes = max(y_test) + 1
config = {'num_layers':3, # number of hidden LSTM layers
          'hidden_size':120, # number of units in each layer
          'grad_max_abs':5, # cutoff for gradient clipping
          'learning_rate':0.0008,
          'classes':num_classes,
          'dropout_keep_prob':dropout,
          'sl':X_train.shape[1],
          'logdir': logdir
        }

"""Create a new model object (construction phase)"""
model = Model(config)
X = model.input
y = model.labels


test_dict = {X:X_test,y:y_test,model.seq_length:test_seq}
val_dict = {X:X_val,y:y_val,model.seq_length:val_seq}

print('Computational graph complete!')
#%%
"""Train the model"""
# initialize variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()


with tf.Session() as sess:
    # Initialize variables
    init.run()

    
    # Create initial values to start the loop
    old_validation_loss = 100
    new_validation_loss = 90
    best_val_acc = 0
    best_epoch = 0
    epoch = 0
    # Create training loop that ends when max epochs is reached or validation suffers
    while epoch < max_epochs:# and new_validation_loss <= 0.9 * old_validation_loss:
        epoch += 1
        # Start of new epoch, reset
        epoch_acc = 0
        # Run through each mini-batch once per epoch

        B = create_batches(X_train,y_train,train_seq,batch_size)
        
        for (batch_x,batch_y,batch_seq) in B:
            train_dict = {X:batch_x, y:batch_y,model.seq_length:batch_seq}
            
            # Run training on the batch
            sess.run(model.training_op,feed_dict=train_dict)
            # Assess and add to accuracy count

            epoch_acc += model.accuracy.eval(feed_dict=train_dict) * batch_x.shape[0]
            logits = model.logits.eval(feed_dict=train_dict)
            epoch_loss += model.loss.eval(feed_dict=train_dict)
            
        # Check to see if this epoch is the best performing one yet
        current_val_acc = model.accuracy.eval(feed_dict=val_dict)
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            best_epoch = epoch
            save_to = saver.save(sess, save_path)
        # Assess validation loss every so often
        if epoch % val_increment == 0:
            old_validation_loss = new_validation_loss
            new_validation_loss = model.loss.eval(feed_dict=val_dict)
            print('%d | train_acc: %f | train_loss: %f | val_acc: %f | val_loss: %f' %(epoch,epoch_acc/X_train.shape[0],epoch_loss/X_train.shape[0], current_val_acc, new_validation_loss))
        else:
            # not a multiple of the increment, just print training data
            print('%d | train_acc: %f | train_loss: %f | val_acc: %f' %(epoch,epoch_acc/X_train.shape[0],epoch_loss/X_train.shape[0],current_val_acc))
    
        # At this point, training has stopped
        # Create train, test, and summary strings
        train_summary_str = model.trainloss_summary.eval(feed_dict={X:X_train, y:y_train,model.seq_length:train_seq})
        test_summary_str = model.testloss_summary.eval(feed_dict=test_dict)
        val_summary_str = model.valloss_summary.eval(feed_dict=val_dict) 
        
        # Save losses and accuracies to the summary file         
        model.file_writer.add_summary(train_summary_str, epoch)
        model.file_writer.add_summary(test_summary_str, epoch)
        model.file_writer.add_summary(val_summary_str, epoch)

    # Print the reason for stopping
    if new_validation_loss > 0.95 * old_validation_loss:
        print('Model overfitted! Stopped training after epoch %d and will use weights from epoch %d'%(epoch,best_epoch))
    elif epoch >= max_epochs:
        print('Reached max number of epochs')
        print('Best epoch was %d'%(best_epoch))
    else:
        print('not sure why we stopped')
        
    saver.restore(sess,save_path)
    test_prediction = model.predictions.eval(feed_dict=test_dict)
    test_acc = model.accuracy.eval(feed_dict=test_dict)
    test_loss = model.loss.eval(feed_dict=test_dict)
    print('Final accuracy:',test_acc,'Final cost:',test_loss)

    # create confusion matrix to visualize classification errors
    confusion_matrix_array = confusion_matrix(y_test,test_prediction)
    cf_normed = np.array(confusion_matrix_array)/np.sum(confusion_matrix_array) * 100
    width = 5
    height = 5
    plt.figure(figsize=(width,height))
    plt.title("%s \n Test accuracy: %f"%(dataset,test_acc))

    plt.imshow(cf_normed, interpolation='nearest', cmap=plt.cm.Blues)

    plt.colorbar()
    tick_marks = np.arange(num_classes)

    # Make a list of strings of the numbered classes
    LABELS = [str(i+1) for i in range(num_classes)] # Can change later once we have name labels if we want
    plt.xticks(tick_marks, LABELS, rotation=90)
    plt.yticks(tick_marks, LABELS)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    summary_op = tf.summary.image("Confusion_Matrix", image)
    summary = sess.run(summary_op)
    model.file_writer.add_summary(summary)
    model.file_writer.close()
    print(confusion_matrix_array)

