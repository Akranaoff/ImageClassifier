from keras.models import Sequential    #will be used to create pipeline
from keras.layers import Conv2D        #will be used to create the Conv matrix
from keras.layers import MaxPooling2D  #Will be used to create the pooling of the Conv matrix
from keras.layers import Flatten       #To Flatten the end Conv and pooled matrix
from keras.layers import Dense         #To create the neural network



#Declare/create the empty pipeline
classifier = Sequential()

#Step 1 - Start the convolution network by adding the size of filter and activation function
classifier.add(Conv2D(32,(3,3), input_shape=(64,64,3), activation='relu'))   #conv2d (numer_of_filter,(filter_size),input_shape(dimension_of_image, channel RGB = 3), activation_function)

#step 2 - Start Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Add another conv and pooling layer
classifier.add(Conv2D(32,(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#step 3 - Flattening the above output
classifier.add(Flatten())

#step 4 - Create a Full connection network
classifier.add(Dense(units = 128, activation='relu'))
classifier.add(Dense(units = 128, activation='relu'))   #Dense will create a hidden layer with the number of perceptron(units) specified along with the activation funciton

#add final output layer
classifier.add(Dense(units = 4, activation='softmax'))

#Compiling above created both the network ie CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Above compile is used at the end neuron/perceptron to calculate the loss function and optimizer for the back propogation




from keras.preprocessing.image import ImageDataGenerator

#Frist create the training and tsesting dataset


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
#resclae image pixel in from the range of 0-255 ie RGB range to the 0-1
#horizontal flip means reversing columns of pixels

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(r'D:\ineuron\Classification_image_2',
                                             target_size= (64,64),
                                                 batch_size= 32,
                                                 class_mode='categorical')
#target size re-size/shrink the images to the size specified
#batch_size is the number of sample from the flatten array to be taken in one go.

test_set = test_datagen.flow_from_directory(r'D:\ineuron\Classification_image_2',
                                                 target_size= (64,64),
                                                 batch_size= 32,
                                                 class_mode='categorical')


model = classifier.fit_generator(training_set,
                                 steps_per_epoch= 100,
                                 epochs= 6,
                                 validation_data = test_set,
                                 validation_steps= 25)

classifier.save("multi_class_model.h5")
print("Saved model to local disk")

#train_val = training_set.class_indices




