import glob
import os
import shutil
from functools import cmp_to_key
from pickle import dump, load

from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import splitfolders
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


def get_epochs(path):
    extensionless_path = os.path.splitext(path)[0]
    return int(extensionless_path.split('_')[-1])


def compare(file1, file2):
    if get_epochs(file1) < get_epochs(file2):
        return -1
    elif get_epochs(file1) > get_epochs(file2):
        return 1
    else:
        return 0


def get_trained_model(model_dir, split):
    if not os.path.exists(model_dir):
        return None, 0

    pathname = fr"{model_dir}/model_{split}_*.hdf5"
    files = [os.path.basename(x) for x in glob.glob(pathname)]

    if not files:
        return None, 0

    files.sort(key=cmp_to_key(compare))

    model_epochs = get_epochs(files[0])
    model_path = os.path.join(model_dir, files[0])

    return keras.models.load_model(model_path), model_epochs


def get_history(history_dir, split):
    if not os.path.exists(history_dir):
        return None

    pathname = fr"{history_dir}/history_{split}_*.txt"
    files = [os.path.basename(x) for x in glob.glob(pathname)]

    if not files:
        return None

    files.sort(key=cmp_to_key(compare))

    history_path = os.path.join(history_dir, files[0])

    with open(history_path, 'rb') as handle:
        history = load(handle)
    return history


def get_model(model_dir, split, img_width, img_height):
    model, epochs = get_trained_model(model_dir, split)
    if model is not None:
        return model, epochs

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

    return model, 0


def get_split1_data():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    test_datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    return train_datagen, test_datagen


def get_split2_data():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
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
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest',
        brightness_range=[0.2, 1.0]
    )

    return train_datagen, test_datagen


def get_split3_data():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
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
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest',
        brightness_range=[0.2, 1.0]
    )

    return train_datagen, test_datagen


def get_split_data(split):
    split_translation = {"1": get_split1_data,
                         "2": get_split2_data,
                         "3": get_split3_data}

    result = split_translation.get(split, None)
    if result is None:
        raise Exception("Wrong split given!")
    train_datagen, test_datagen = result()

    return train_datagen, test_datagen


def train(train_data_dir, test_data_dir, model_dir, history_dir, split):
    img_width, img_height = 224, 224
    nb_train_samples = 16324
    nb_validation_samples = 1819
    epochs = 200
    batch_size = 16

    try:
        train_datagen, test_datagen = get_split_data(split)
    except Exception as e:
        print(e)
        return

    model, epochs_already_done = get_model(model_dir, split, img_width, img_height)
    old_history = get_history(history_dir, split)

    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size, class_mode='categorical',
                                                        shuffle=True)

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size, class_mode='categorical', shuffle=True)

    history = model.fit(train_generator,
                        steps_per_epoch=nb_train_samples // batch_size,
                        epochs=epochs, validation_data=test_generator,
                        validation_steps=nb_validation_samples // batch_size,
                        verbose=1)

    merged_history = merge_history(history.history, old_history)

    new_epochs = epochs_already_done + epochs
    plot_graphs(merged_history, split, new_epochs)

    save(model, model_dir, merged_history, history_dir, split, new_epochs)


def save(model, model_dir, history, history_dir, split, epochs):
    filename_model = f"model_{split}_{epochs}.hdf5"
    path = os.path.join(model_dir, filename_model)
    model.save(path)

    filename_history = f"history_{split}_{epochs}.txt"
    path = os.path.join(history_dir, filename_history)
    with open(path, 'wb') as handle:
        dump(history, handle)


def merge_history(new_history, old_history):
    if old_history is None:
        old_history = {'loss': [],
                       'accuracy': [],
                       'val_loss': [],
                       'val_accuracy': []}

    old_history['loss'].extend(new_history['loss'])
    old_history['accuracy'].extend(new_history['accuracy'])
    old_history['val_loss'].extend(new_history['val_loss'])
    old_history['val_accuracy'].extend(new_history['val_accuracy'])

    return old_history


def plot_graphs(history, split, epoch):
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    img_name = f"model_accuracy_split{split}_epoch{epoch}.png"
    plt.savefig(img_name)

    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    img_name = f"model_loss_split{split}_epoch{epoch}.png"
    plt.savefig(img_name)


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


def split_folders(input_dir, output_dir):
    splitfolders.ratio(input_dir, output=output_dir, seed=1338,
                       ratio=(.8, .1, .1), group_prefix=None)  # default values


def prepare_dataset(traffic_lights_dir, classified_traffic_lights_dir, split_dataset_dir):
    if is_dataset_prepared(split_dataset_dir):
        print("Dataset is already prepared. Skipping...")
        return
    split_traffic_lights_into_subfolders(traffic_lights_dir, classified_traffic_lights_dir)
    split_folders(classified_traffic_lights_dir, split_dataset_dir)


def is_dataset_prepared(split_dataset_dir):
    if os.path.exists(split_dataset_dir) and os.path.isdir(split_dataset_dir):
        if not os.listdir(split_dataset_dir):
            return False
        else:
            return True
    else:
        return False


def main():
    # Adjust paths to your system
    traffic_lights_dir = r"Y:\DTLD\TraficLightsShuffled"
    classified_traffic_lights_dir = r"Y:\DTLD\TraficLightsShuffled\Input"
    split_dataset_dir = r"D:\Szum\SZuM3\Output"

    train_data_dir = os.path.join(split_dataset_dir, "train")
    test_data_dir = os.path.join(split_dataset_dir, "test")

    model_dir = r"D:\Szum\SZuM3\lights-detection\models"
    history_dir = r"D:\Szum\SZuM3\lights-detection\history"

    split = "2"  # Change to your split
    prepare_dataset(traffic_lights_dir, classified_traffic_lights_dir, split_dataset_dir)
    train(train_data_dir, test_data_dir, model_dir, history_dir, split)


if __name__ == "__main__":
    main()
