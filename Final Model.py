#!/usr/bin/env python
# coding: utf-8

# ## Final Model

# In[194]:


from tensorflow.keras.layers import Input, LSTM, Dense, Reshape, Conv2D, Flatten, Concatenate, TimeDistributed,BatchNormalization, LayerNormalization,LeakyReLU
from tensorflow.keras.models import Model
from time import time
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
#!pip install opencv-python
import cv2
from tensorflow.keras.losses import CategoricalCrossentropy
#!pip install natsort
from natsort import natsorted
import os
import time
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define directories
train_dir = 'C:/Users/mahesh.inamdar/Desktop/codes/run/Thick/WLD_Scharr/train'
validation_dir = 'C:/Users/mahesh.inamdar/Desktop/codes/run/Thick/WLD_Scharr/val'
test_dir = 'C:/Users/mahesh.inamdar/Desktop/codes/run/Thick/WLD_Scharr/test'

class TemporalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, units, spatial_units, **kwargs):
        super(TemporalSelfAttention, self).__init__(**kwargs)
        self.units = units
        self.spatial_units = spatial_units
        self.Dense_temp=tf.keras.layers.Dense(units=self.units)
        self.Spa_temp=tf.keras.layers.Dense(units=self.units)

    def build(self, input_shape):
        self.W_query_temporal = self.add_weight(shape=(input_shape[-1], self.units),
                                                initializer='glorot_uniform',
                                                trainable=True)
        self.W_key_temporal = self.add_weight(shape=(input_shape[-1], self.units),
                                              initializer='glorot_uniform',
                                              trainable=True)
        self.W_value_temporal = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                                initializer='glorot_uniform',
                                                trainable=True)

        self.W_query_spatial = self.add_weight(shape=(input_shape[-1], self.spatial_units),
                                               initializer='glorot_uniform',
                                               trainable=True)
        self.W_key_spatial = self.add_weight(shape=(input_shape[-1], self.spatial_units),
                                             initializer='glorot_uniform',
                                             trainable=True)
        self.W_value_spatial = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                               initializer='glorot_uniform',
                                               trainable=True)

        super(TemporalSelfAttention, self).build(input_shape)

    def call(self, inputs):
        temporal_query = tf.matmul(inputs, self.W_query_temporal)
        temporal_key = tf.matmul(inputs, self.W_key_temporal)
        temporal_value = tf.matmul(inputs, self.W_value_temporal)

        spatial_query = tf.matmul(inputs, self.W_query_spatial)
        spatial_key = tf.matmul(inputs, self.W_key_spatial)
        spatial_value = tf.matmul(inputs, self.W_value_spatial)

        # Temporal attention
        temporal_attention_scores = tf.matmul(temporal_query, temporal_key, transpose_b=True)
        temporal_attention_weights = tf.nn.softmax(temporal_attention_scores / tf.sqrt(tf.cast(tf.shape(temporal_key)[-1], tf.float32)), axis=-1)
        temporal_attention_output = tf.matmul(temporal_attention_weights, temporal_value)

        # Spatial attention
        spatial_attention_scores = tf.matmul(spatial_query, spatial_key, transpose_b=True)
        spatial_attention_weights = tf.nn.softmax(spatial_attention_scores / tf.sqrt(tf.cast(tf.shape(spatial_key)[-1], tf.float32)), axis=-1)
        spatial_attention_output = tf.matmul(spatial_attention_weights, spatial_value)

        # Channel-wise fusion
        temporal_channel = self.Dense_temp(temporal_attention_output)
        spatial_channel = self.Spa_temp(spatial_attention_output)
        # Joint attention fusion
        joint_attention_ = tf.keras.layers.Add()([temporal_channel, spatial_channel])
        joint_attention = tf.keras.layers.Activation('sigmoid')(joint_attention_)  # Apply sigmoid activation to emphasize informative features
        # Apply joint attention to inputs
        joint_attention_output = tf.keras.layers.Multiply()([inputs, joint_attention])

        return joint_attention_output

    def compute_output_shape(self, input_shape):
        return input_shape

# Define input shape for images
image_height = 128    # Height of input images
image_width = 128     # Width of input images
channels = 3          # Number of image channels (e.g., RGB)
num_classes = 3       # Number of output classes
patch_size = (128, 128)  # Patch size
batch_size=3

# Define input tensor for sequence of images
max_sequence_length = 16  # Set your desired maximum sequence length
input_shape = (max_sequence_length,image_height, image_width, channels)
input_images = Input(shape=input_shape)
leaky_relu = LeakyReLU(alpha=0.2)
# Apply Temporal Self-Attention with Joint Attention Fusion
conv_layer1 = TimeDistributed(Conv2D(16, kernel_size=(3, 3)))(input_images)
activated_conv_layer1 = TimeDistributed(leaky_relu)(conv_layer1)
BN1=TimeDistributed(BatchNormalization()) (activated_conv_layer1)
conv_layer2 = TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu'))(BN1)
activated_conv_layer2 = TimeDistributed(leaky_relu)(conv_layer2)
LN1=TimeDistributed(LayerNormalization()) (activated_conv_layer2)

temporal_attention_output = TimeDistributed(TemporalSelfAttention(units=32, spatial_units=32))(conv_layer2)
Joint_attention_output = tf.keras.layers.Multiply()([conv_layer2, temporal_attention_output])

# Flatten and apply Dense layers
flattened_output = TimeDistributed(Flatten())(Joint_attention_output)
dense_layer1 = TimeDistributed(Dense(32, activation='tanh'))(flattened_output)
output = TimeDistributed(Dense(3, activation='softmax'))(dense_layer1)

# Create model
model = Model(inputs=input_images, outputs=output)

# Compile model
model.compile(optimizer='RMSprop', loss=SparseCategoricalCrossentropy(), metrics= tf.keras.metrics.SparseCategoricalAccuracy())

# Print model summary
model.summary()

def resize_image(image, target_size=(512, 512)):
    resized_image = cv2.resize(image, target_size)
    return resized_image

def extract_patches(image, patch_size):
    patches = []
    height, width = image.shape[:2]
    for y in range(0, height - patch_size[0] + 1, patch_size[0]):
        for x in range(0, width - patch_size[1] + 1, patch_size[1]):
            patch = image[y:y+patch_size[0], x:x+patch_size[1]]
            patches.append(patch)
    return patches

# Example usage:
def load_images_from_directory(directory):
    images = []
    for filename in natsorted(os.listdir(directory)):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = (os.path.join(directory, filename))
            image = augment_image(resize_image(cv2.imread(image_path)))
            image=extract_patches(image,(128,128))
            if image is not None:
                images.append(image)
    return np.array(images)

def get_class_label(patient_names):
    labels=[]
    for names in patient_names:
        if 'acute' in names:
            labels.append(0)
        elif 'chronic' in names:
            labels.append(1)
        elif 'normal' in names:
            labels.append(2)
    return np.array(labels)

# Function to extract patches from an image
def extract_patches(image, patch_size):
    patches = []
    height, width = image.shape[:2]
    for y in range(0, height - patch_size[0] + 1, patch_size[0]):
        for x in range(0, width - patch_size[1] + 1, patch_size[1]):
            patch = image[y:y+patch_size[0], x:x+patch_size[1]]
            patches.append(patch)
    return patches

def augment_image(image):
    if np.random.rand() < 0.5:
        image_np = np.fliplr(image)
        rotations = [0, 1, 2, 3]  # 0, 90, 180, 270 degrees
        rotation_idx = np.random.choice(rotations)
        image = np.rot90(image_np, k=rotation_idx)
    return np.array(image)

def generate_patient_data(directory):
    for folders in os.listdir(directory):
        for patient in natsorted(os.listdir(directory+'/'+folders)):
            #print(folders,patient)
            patient_names=os.listdir(directory+'/'+folders+'/'+patient)
            patient_images=load_images_from_directory(directory+'/'+folders+'/'+patient)
            patient_labels=get_class_label(patient_names)
            #print(len(patient_labels),patient_labels)
            reshaped_labels = patient_labels.reshape(-1, 1, 1)
            reshaped_labels = np.repeat(reshaped_labels, repeats=16, axis=1)
            for i in range(len(os.listdir(directory+'/'+folders+'/'+patient))//batch_size):
                P=patient_images[i * batch_size: (i + 1) * batch_size]
                L=patient_labels[i * batch_size: (i + 1) * batch_size].reshape(-1,1,1)
                yield  np.array(P), np.array(np.repeat(L,repeats=16, axis=1))

initial_learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

# Define learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True)
max_grad_norm = 1.0

epoch_loss_list = []
epoch_accuracy_list = []
epoch_recall_list = []
epoch_precision_list = []
epoch_f1_list = []

accuracy_metric =tf.keras.metrics.SparseCategoricalAccuracy()
recall_metric = tf.keras.metrics.Recall()
precision_metric = tf.keras.metrics.Precision()

with tf.device('/CPU:0'):
    for epoch in range(10):
        for batch_images, batch_labels in generate_patient_data(train_dir):
          # Specify the GPU device
             with tf.GradientTape() as tape:
                    # Forward pass
                    predictions = model(batch_images, training=True)
                    # Compute loss
                    loss = model.loss(batch_labels, predictions)
                    # Compute gradients
                    gradients = tape.gradient(loss, model.trainable_variables)
                            # Clip gradients
                    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_grad_norm)
                            # Update model weights
                    optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
                        #time.sleep(2)
                        #for grad in gradients:
                        #    print('gradients',tf.reduce_mean(grad))
                    #for batch_images, batch_labels in generate_patient_data(train_dir):  # Assuming you have a separate data generator for training
                        #batch_predictions = model.predict_on_batch(batch_images)
                    accuracy_metric.update_state(batch_labels, predictions)
                    recall_metric.update_state(batch_labels, predictions[:,:,0])
                    precision_metric.update_state(batch_labels, predictions[:,:,0])
                    
                # Calculate metrics for the epoch
        epoch_loss = loss.numpy()  # Assuming loss is a scalar Tensor
        epoch_accuracy = accuracy_metric.result().numpy()
        epoch_recall = recall_metric.result().numpy()
        epoch_precision = precision_metric.result().numpy()
        epoch_f1 = np.multiply(2,np.multiply(epoch_precision,epoch_recall)) / (epoch_precision + epoch_recall + 1e-7)  # Compute F1 score

        print(f"Epoch {epoch+1}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}, Recall: {epoch_recall}, Precision: {epoch_precision}, F1-score: {epoch_f1}")
                    
            # Store epoch-level metrics in lists
        epoch_loss_list.append(epoch_loss)
        epoch_accuracy_list.append(epoch_accuracy)
        epoch_recall_list.append(epoch_recall)
        epoch_precision_list.append(epoch_precision)
        epoch_f1_list.append(epoch_f1)
                    
        accuracy_metric.reset_states()
        recall_metric.reset_states()
        precision_metric.reset_states()

                    # Update learning rate
        optimizer.learning_rate = lr_schedule(epoch)
                    # Print loss and accuracy after each epoch
                    #print(f"Epoch {epoch+1}, Loss: {loss}, Accuracy: {model.evaluate(batch_images, batch_labels, verbose=0)[1]}")
            
    
                #---------TEST--------------------
                # Generate patient-wise predictions
    
    patient_predictions = []
    patient_accuracy_list = []
    patient_recall_list = []
    patient_precision_list = []
    patient_f1_list = []

    for batch_images, batch_labels in generate_patient_data(test_dir):
        batch_predictions = model.predict_on_batch(batch_images)
        patient_prediction = np.mean(batch_predictions, axis=0)  # Aggregate predictions for the patient
        patient_predictions.append(patient_prediction)

                    # Update metrics with current batch predictions
        patient_predictions_ = np.array(patient_predictions)
                    #reshaped_batch_labels=np.reshape(batch_labels,(16,3))
        accuracy_metric.update_state(batch_labels, np.reshape(patient_prediction,(3,16,1)))
        recall_metric.update_state(batch_labels, np.reshape(patient_prediction,(3,16,1)))
        precision_metric.update_state(batch_labels, np.reshape(patient_prediction,(3,16,1)))

                # Calculate patient-level metrics
        patient_accuracy = accuracy_metric.result().numpy()
        patient_recall = recall_metric.result().numpy()
        patient_precision = precision_metric.result().numpy()
        patient_f1 = np.multiply(2,np.multiply(patient_precision,patient_recall)) / (patient_precision + patient_recall + 1e-7)  # Compute F1 score

                # Print or log patient-level metrics
        print('TEST',f"Patient Accuracy: {patient_accuracy}, Patient Recall: {patient_recall}, Patient Precision: {patient_precision}, Patient F1-score: {patient_f1}")

                # Reset metrics for next evaluation
        accuracy_metric.reset_states()
        recall_metric.reset_states()
        precision_metric.reset_states()

# In[175]:




# In[1]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

