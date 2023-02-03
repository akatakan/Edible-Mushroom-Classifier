from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras import Sequential
import tensorflow as tf





TRAINING_DIR = r"C:\Users\Atakan\Desktop\Edible-Mushroom-Classification-main\mantarlar\train"
TEST_DIR = r"C:\Users\Atakan\Desktop\Edible-Mushroom-Classification-main\mantarlar\test"
VALIDATION_DIR = r"C:\Users\Atakan\Desktop\Edible-Mushroom-Classification-main\mantarlar\validation"

train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    rotation_range=45,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    brightness_range=[0.4,1.5],
                                    fill_mode="nearest")
test_datagen = ImageDataGenerator(
    rescale=1./255     
)

val_datagen= ImageDataGenerator(
    rescale=1./255
)


training_set = train_datagen.flow_from_directory(TRAINING_DIR,
                                        target_size=(224, 224),
                                        batch_size=21,
                                        class_mode='binary',
                                        shuffle=True)
test_set= test_datagen.flow_from_directory(TEST_DIR,
                                        target_size=(224, 224),
                                        batch_size=1,
                                        class_mode='binary',
                                        shuffle=True)

validation_set = val_datagen.flow_from_directory(VALIDATION_DIR,
                                        target_size=(224, 224),
                                        batch_size=1,
                                        class_mode='binary',
                                        shuffle=True)

total_train=674
total_validation=85
batch_size=32

model = Sequential()

model.add(Conv2D(32,
          input_shape=(224,224,3),
          kernel_size=(3,3),
          padding="same",
          activation="relu"
                  ))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,
          kernel_size=(3,3),
          padding="same",
          activation="relu"
                  ))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128,
          kernel_size=(3,3),
          padding="same",
          activation="relu"
                  ))
model.add(MaxPooling2D((2,2)))


model.add(Flatten())

model.add(Dense(64,activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(32,activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(2,activation="softmax"))

model.compile(optimizer=Adam(),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])



results = model.fit(
    training_set,
    steps_per_epoch=(total_train//batch_size),
    epochs = 200,
    validation_data=validation_set,
    validation_steps=(total_validation//batch_size),
    batch_size = batch_size,
    verbose = 1
)

loss, accuracy = model.evaluate(test_set)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model1.tflite","wb") as f:
    f.write(tflite_model)
  