# Enhancing Vision with Convolutional Neural Networks

## What are convolutions and pooling?
* Some convolutions will change the image in such a way that certain features in the image get emphasized
*  Pooling is a way of compressing an image

```python
model = tf.keras.models.Sequential([
    # input layer in the shape of our data
    tf.keras.layers.Flatten(), # input_shape=(28,28)
    # hidden layer
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    # output layer in the shape of the number categories
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
```

## Implementing convolutional layers

```python
model = tf.keras..models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### Conv2D
```python
tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1))
```

### MaxPooling2D
```python
tf.keras.layers.MaxPooling2D(2,2)
```

* In max-pooling, we're going to take the maximum value
* It's a two-by-two pool, so for every four pixels, the biggest one will **survive** 

Then, we add another convolutional later and another max-pooling layer and then again, pool to reduce the size. So, by the time the image gets to the flattern to go into the dense layers, it's already **much smaller**.

```python
tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
tf.keras.layers.MaxPooling2D(2,2),
```

So, its content has been greatly simplified, the goal being that the convolutions will filter it to the features that determine the output.

```python
model.summary()
```

## Other Notes
**Overfitting** occurs when the network learns the data from the training set really well, but it's too specialised to only that data, and as a result is less effective at interpreting other unseen data. For example, if all your life you only saw red shoes, then when you see a red shoe you would be very good at identifying it. But blue suede shoes might confuse you.

## Ungraded Lab
* Lab 1: [Improving accuracy using convolution](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W3/ungraded_labs/C1_W3_Lab_1_improving_accuracy_using_convolutions.ipynb)
* Lab 2: [Exploring Convolutions](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W3/ungraded_labs/C1_W3_Lab_2_exploring_convolutions.ipynb)

## Graded Assignments
* Quiz 3: [week-3-quiz](Graded%20Assignment/week3-quiz.md)
* Programming assignment: [notebook](Graded%20Assignment/)

## Reference
* [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)
* [MaxPooling2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D)
* [Convolution Neural Network - Youtube](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF)
* [Image Filtering](https://lodev.org/cgtutor/filtering.html)
