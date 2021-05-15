import os
import shutil

from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import splitfolders
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


def main():
    img_width, img_height = 224, 224

    train_data_dir = 'output/train'
    validation_data_dir = 'output/test'
    nb_train_samples = 16324
    nb_validation_samples = 1819
    epochs = 20
    batch_size = 16

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (5, 5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(5))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest',
        brightness_range=[0.2, 1.0]
    )

    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest',
        brightness_range=[0.2, 1.0]
    )

    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size, class_mode='categorical',
                                                        shuffle=True)

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size, class_mode='categorical', shuffle=True)

    history = model.fit(train_generator,
                        steps_per_epoch=nb_train_samples // batch_size,
                        epochs=epochs, validation_data=validation_generator,
                        validation_steps=nb_validation_samples // batch_size,
                        verbose=1)
    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.show()
    model.save_weights('model1_saved.h5')


def get_colour(path):
    return path.split('-')[-2]


def split_traffic_lights_into_subfolders(traffic_lights_dir, output_dir):
    folder_translation = {"0": "None0",
                          "1": "Red1",
                          "2": "Yellow2",
                          "3": "Red_Yellow3",
                          "4": "Green4"}

    for subfolder in folder_translation.values():
        directory = os.path.join(output_dir, subfolder)
        if not os.path.exists(directory):
            os.makedirs(directory)

    for root, dirs, files in os.walk(traffic_lights_dir):
        for file in files:
            colour = get_colour(file)
            subfolder = folder_translation.get(colour)

            src = os.path.join(traffic_lights_dir, file)
            dst = os.path.join(output_dir, subfolder, file)
            shutil.copyfile(src, dst)


def split_folders():
    images_path = r"Y:\DTLD\TraficLightsShuffled\Input"
    output = r"D:\Szum\SZuM3\Output"
    splitfolders.ratio(images_path, output=output, seed=1338,
                       ratio=(.8, .1, .1), group_prefix=None)  # default values


def prepare_files():
    traffic_lights_dir = r"Y:\DTLD\TraficLightsShuffled"
    output_dir = r"Y:\DTLD\TraficLightsShuffled\Input"
    split_traffic_lights_into_subfolders(traffic_lights_dir, output_dir)


def prepare_dataset():
    prepare_files()
    split_folders()


if __name__ == "__main__":
    # Do it only once
    prepare_dataset()

    # main()
