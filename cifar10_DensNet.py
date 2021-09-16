from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, AveragePooling2D, Input, Flatten, concatenate, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical, plot_model
import os
import numpy as np

batch_size = 200
epochs = 50
num_classes = 10

num_filters = 12
num_dense_blocks = 3
num_bottleneck_layers = 2

# load the cifar10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# input image dimensions
input_shape = x_train.shape[1:]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

def lr_schedule(epoch):
    """
    Learning rate is reduced after 10, 20, 60, 80 epochs.
    Called automatically every epoch as part of callbacks.
    """
    lr = 1e-3
    if epoch > 80:
        lr *= 0.5e-3
    elif epoch > 60:
        lr *= 1e-3
    elif epoch > 20:
        lr *= 1e-2
    elif epoch > 10:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def Densnet_layers(inputs, num_filters=num_filters, activation='relu', kernel_size = 3, AveragePooling = None):

    x = BatchNormalization()(inputs)
    if activation is not None:
        x = Activation(activation)(x)
    x = Conv2D(num_filters,
        kernel_size=kernel_size,
        padding='same',
        kernel_initializer='he_normal')(x)
    if AveragePooling is not None:
        x = AveragePooling2D()(x)

    return x

def DensNet(input_shape):
    
    # BN-ReLU-Conv2D densenet CNNs
    inputs = Input(shape=input_shape)
    x = Densnet_layers(inputs)
    x = concatenate([inputs, x])

    # stack of dense blocks bridged by transition layers
    for i in range(num_dense_blocks): 
        # a dense block is a stack of bottleneck layers
        for j in range(num_bottleneck_layers): 
            y = Densnet_layers(x, num_filters=2 * num_filters, kernel_size=1)
            y = Densnet_layers(y, num_filters=4 * num_filters)
            x = concatenate([x, y])

        # not to include transition layer after the last dense block
        if i == num_dense_blocks - 1:
            continue

        x = Densnet_layers(x, num_filters=num_filters, kernel_size=1, activation=None, AveragePooling=True)

    # AveragePooling2D is used to reduce the size of feature map to 1 x 1
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    kernel_initializer='he_normal',
                    activation='softmax')(y)
    model = Model(inputs=inputs, outputs=outputs)

    return model

model = DensNet(input_shape)
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(1e-3),
              metrics=['acc'])
model.summary()
plot_model(model, to_file="cifar10-densenet.png", show_shapes=True)

# saving the model directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_densenet_model.{epoch:02d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# callbacks for model saving and learning rate reducer
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

call_backs = [checkpoint, lr_scheduler]

model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            shuffle=True,
            callbacks=call_backs)

scores = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
print("\nTesting accuracy: %.1f%%" % (100.0 * scores[1]))