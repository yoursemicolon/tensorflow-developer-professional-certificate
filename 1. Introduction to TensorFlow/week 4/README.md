# Using Real-world Images

## Understanding ImageDataGenerator
```python
import tensorflow as tf
from tf.keras.preprocessing.image import ImageDataGenerator

# instantiate an image generator
# pass rescale to normalize the data
train_datagen = ImageDataGenerator(rescale=1./255)

# point out directory that contains sub-directories that contain images
# the name of sub-directories will be the labels for the images
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(300, 300), # input data all has to be the same size
    batch_size=128,
    class_mode='binary' # binary classifier
)

test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary'
)
```

* The images are resized as they are loaded so we don't need to do preprocessing
* Batch size is a term used in machine learning and refers to the number of training examples utilized in one iteration.

## Defining a ConvNet to use complex images
Here, we use three sets of convolution pooling layers.
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

In input shape, because the images are in color we use 3 bytes per pixel for red, green and blue.
```python
tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
```

For the output layer, we have one neuron for two classes. It because we use different activation. Sigmoid is great for binary classification, where one class will tend towards zero and the other class tending towards one. We could use two neurons here too, and the same softmax function as before, but for binary this is a bit more efficient. 
```python
tf.keras.layers.Dense(1, activation='sigmoid')
```

## Training The ConvNet
```python
from tensorflow.keras.optimizers import RMSprop

model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=0.001),
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=8,
    verbose=2
)
```

* There are 1,024 images in the training directory, we're loading them in 128 at a time. In order to load them all, we need to do 8 batches so we set **steps_per_epoch** to cover that.
* We have 256 images from validation_genertaor and we wanted to handle them in batches of 32, so we will do 8 steps.
* Verbose specifies how much to display while training is going on. With verbose set to 2, we'll get a little less animation hiding the epoch progress.

```python
import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():

    # predicting images
    path = '/content/' + fn
    img = image.load_img(path, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes[0])
    if classes[0]>0.5:
        print(fn + " is a human")
    else:
        print(fn + " is a horse")
```

This give us the button that can be pressed to pick one or more images to upload.
```python
...
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
    ...
    ...
```

The loop then iterates through all of the images in that collection.
```python
 img = image.load_img(path, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
```




