# Recurrent Neural Networks for Time Series

Recurrent Neural networks (RNN) and Long Short Term Memory (LSTM) networks are really useful to classify and predict on sequential data. 

### Sequence Data
Sequential Data is any kind of data where the order matters as you said. So we can assume that time series is a kind of sequential data, because the order matters.

### Sequence Model
Sequence models are the machine learning models that input or output sequences of data. Sequential data includes text streams, audio clips, video clips, time-series data and etc.

### Lambda Layers
Lambda layers allow us to write effectively an arbitrary piece of code as a layer in the neural network. Basically a Lambda function, an unnamed function, but implemented as a layer in the neural network that resend the data, scales it. More simply we can say that using the lambda layer we can transform the data before applying that data to any of the existing layers.

## Conceptual Overview
One difference will be that the full input shape when using RNNs is three-dimensional. The first dimension will be the batch size, the second will be the timestamps, and the third is the dimensionality of the inputs at each time step. 

<p align="center">
    <img src="img\recurrent-layer.PNG" alt="metrics" width=500>
</p>

What it looks like there's lots of cells, there's actually only one, and it's used repeatedly to compute the outputs. This is what gives this type of architecture the name a recurrent neural network, because the values recur due to the output of the cell, a one-step being fed back into itself at the next time step.

## When to Use RNN and LSTM
RNNs are particularly suited for tasks that involve sequences (thanks to the recurrent connections). For example, they are often used for machine translation, where the sequences are sentences or words. In practice, an LSTM is often used, as opposed to a vanilla (or standard) RNN, because it is more computationally effective. In fact, the LSTM was introduced to solve a problem that standard RNNs suffer from, i.e. the vanishing gradient problem. (Now, for these tasks, there are also the transformers, but the question was not about them).

## References
* [Sequence Models & Recurrent Neural Networks](https://towardsdatascience.com/sequence-models-and-recurrent-neural-networks-rnns-62cadeb4f1e1#:~:text=Sequence%20models%20are%20the%20machine,algorithm%20used%20in%20sequence%20models.&text=1.)
* [Introduction to Sequence Modeling Problems](https://towardsdatascience.com/introduction-to-sequence-modeling-problems-665817b7e583)
* [Ungraded Lab: Using a Simple RNN for forecasting](https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/main/C4/W3/ungraded_labs/C4_W3_Lab_1_RNN.ipynb)
* [Ungraded Lab: Using a multi-layer LSTM for forecasting](https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/main/C4/W3/ungraded_labs/C4_W3_Lab_2_LSTM.ipynb)
* [Huber Loss](https://en.wikipedia.org/wiki/Huber_loss)
* [LSTM Lesson](https://www.coursera.org/lecture/nlp-sequence-models/long-short-term-memory-lstm-KXoay)