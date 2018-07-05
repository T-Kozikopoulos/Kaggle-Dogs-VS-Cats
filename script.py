# Kaggle competition: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/leaderboard

# For working with images.
import cv2
# For working with arrays.
import numpy as np
# For working with directories
import os
# For shuffling data.
from random import shuffle
# Progress bar for how long until a piece code is done running.
from tqdm import tqdm
# For visualization.
import matplotlib.pyplot as plt
# The imports below are for creating the Convolutional Neural Network.
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# Change directory to where you downloaded the data.
TRAIN_DIR = 'C:/Users/a/Desktop/Dogs Vs Cats/train'
TEST_DIR = 'C:/Users/a/Desktop/Dogs Vs Cats/test'
# This is what we're gonna resize all the images to.
IMG_SIZE = 50
# Learning rate(=0.001).
LR = 1e-3

# For training and saving/loading different models.
MODEL_NAME = 'dogsVScats_{}_{}.model'.format(LR, 'final')


# Creating the labels.
def label_img(img):
    # The images are named 'cat.0', 'dog.0', 'cat.1' etc.
    # So, we get our labels from the file names, minus the last 4 characters(file extension '.jpg').
    word_label = img.split('.')[-3]
    # In the competition submission, cats are labeled 1 and dogs 0.
    if word_label == 'cat':
        return [1, 0]
    elif word_label == 'dog':
            return [0, 1]


# Creating the features.
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        # Resize images and turn into grayscale to make processing them lighter.
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        # Collect all the data in one place.
        training_data.append([np.array(img), np.array(label)])

    # Randomize the data otherwise all the cats will be at the start and the dogs at the end.
    shuffle(training_data)
    # Save once, so we only load the data and not have to run the function again in the future.
    np.save('train_data.npy', training_data)
    return training_data


# Almost the same as above, but not exactly because in this case the data isn't labeled.
def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

# Run 'train_data = create_train_data()' only once or if you make changes to the data.
# Otherwise, comment it out and use this to load the data much faster instead.
train_data = np.load('train_data.npy')

# Creating the CNN. First, the input layer.
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

# Creating a layer.
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

# Adding a second one.
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

# And then a bunch more.
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

# Creating a fully connected layer.
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

# Adding the final part, the output layer.
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

# Creating the model.
model = tflearn.DNN(convnet, tensorboard_dir='log')

#  Check if the trained model already exists, instead of training a new one even though we might not have to.
if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('Model loaded.')

# Reserving a portion of the training data for testing while training the model, so we can see our model's accuracy.
train = train_data[:-500]
test = train_data[-500:]

# Getting our features and labels in the proper form.
X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = [i[1] for i in train]

test_X = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]

# Fitting the model. 13 epochs worked fine for me without overfitting.
model.fit({'input': X}, {'targets': y}, n_epoch=13, validation_set=({'input': test_X}, {'targets': test_y}),
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

# Save model for re-use or more training later.
model.save(MODEL_NAME)

# Create the test data.
test_data = process_test_data()

# Visualize to make sure everything is OK.
fig = plt.figure()

for num, data in enumerate(test_data[:12]):
    # Again, cat=[1,0] and dog=[0,1]
    img_num = data[1]
    img_data = data[0]

    # 'num + 1' because subplots start from 1 not 0
    y = fig.add_subplot(3, 4, num + 1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)

    # Use the model to predict the labels on the test data.
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 0:
        str_label = 'Cat'
    else:
        str_label = 'Dog'

    # Gotta use 'gray' despite turning the image to grayscale before.
    y.imshow(orig, cmap='gray')

    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.show()

# Save the predictions into a submission file of the specified format
with open('submission_file.csv', 'w') as f:
    f.write('id,label\n')

with open('submission_file.csv', 'a') as f:
    for data in tqdm(test_data):
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
        model_out = model.predict([data])[0]
        f.write('{},{}\n'.format(img_num, model_out[1]))
