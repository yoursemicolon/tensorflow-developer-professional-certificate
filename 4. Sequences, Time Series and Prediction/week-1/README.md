# Sequences and Prediction

Take a look at some of the unique considerations involved when handling sequential time series data -- where values change over time, like the temperature on a particular day, or the number of visitors to your web site. We'll discuss various methodologies for predicting future values in these time series, building on what you've learned in previous courses!

## Introduction
Time-series is one part of sequence models where it's a case of if you can imagine a series of data that changes over time. It might be the closing prices for stock on the stock exchange, or it could be weather. It could be how sunny it is in California on a given day, or how rainy it is in Seattle on a given day, that type of thing. So if you just imagine how an item of data changes over time and how it's measured over time.

We're going to start by creating a synthetic sequence of data, so that we can start looking at what the common attributes that you see in data series are. So for example:
* Data can be seasonal. It's sunnier in June than it is in January or it's wetter in November than it is in October, something along those lines. So you have that seasonality of data. 
* Data can have trends, like whether it probably doesn't really trend although we could argue that it strangely enough idea with climate change, but like a stock data may trend upwards over time or downwards over some other times, and then of course the random factor that makes it hard to predict is noise. 

So you can have like seasonal data, you can have trends in your data, but then you can have noise on that data as well.

## Time Series Example
Different type of time series, looking at basic forecasting around them. Time series are everywhere. You may have seen them in stock prices, weather forecasts, historical trends, such as Moore's law. What exactly is a time series? It's typically defined as an ordered sequence of values that are usually equally spaced over time. So for example, every year in my Moore's law charts or every day in the weather forecast. 

In each of these examples, there is a single value at each time step, and as a results, the term **univariate** is used to describe them. You may also encounter time series that have multiple values at each time step. As you might expect, they're called **Multivariate Time Series**. 

Multivariate Time Series charts can be useful ways of understanding the impact of related data. For example, consider this chart of births versus deaths in Japan from 1950 to 2008. It clearly shows the two converging, but then deaths begin to outstrip births leading to a population decline. Now, while they could be treated as two separate univariate time series, the real value of the data becomes apparent when we show them together as a multivariate. 

<p align="center">
    <img src="img\birth-and-date.PNG" alt="sequences" width=500>
</p>

## Machine learning applied to time series
What can we do?
1. Forecasting based on the data, example: the birth and death rate chart for Japan. It would be very useful to predict future values so that government agencies can plan for retirement, immigration and other societal impacts of these trends.
2. Detect anomalies. For example, in website logs so that you could see potential denial of service attacks showing up as a spike on the time series like this.
3. Analyze the time series to spot patterns in them that determine what generated the series itself. A classic example of this is to analyze sound waves to spot words in them which can be used as a neural network for speech recognition.

## Common patterns in time series
Time-series come in all shapes and sizes, but there are a number of very common patterns. 
1. Trend, where time series have a specific direction that they're moving in. The general tendency of the values to go up or down as time progresses. 
2. Seasonality, which is seen when patterns repeat at predictable intervals. For instance, the hourly temperature might oscillate similarly for 10 consecutive days and you can use that to predict the behavior on the next day.
3. Auto correlation, measurements at a given time step is a function of previous time steps
4. Noise, not predictable at all and just a complete set of random values producing what's typically called white noise
5. Non-stationary, break an expected pattern. Big events can alter the trend or seasonal behavior of the data, later its behavior does not change over time

We always assume that more data is better. But for **time series forecasting it really depends on the time series**. If it's stationary, meaning its behavior does not change over time, then great. The more data you have the better. But if it's not stationary then the optimal time window that you should use for training will vary. Ideally, we would like to be able to take the whole series into account and generate a prediction for what might happen next.

## Train, validation and test sets
**Naïve forecasting** is the technique in which the last period's sales are used for the next period's forecast without predictions or adjusting the factors.

### **How Do We Measure Performance?**
To measure the performance of our forecasting model,. We typically want to split the time series into a training period, a validation period and a test period — **fixed partitioning.** 

<p align="center">
    <img src="img\fixed-partitioning.PNG" alt="fixed-partitioning" width=500>
</p>

We'll train the model on the training period, and we'll evaluate it on the validation period. And work in it and the hyperparameter, until we get the desired performance, measured using the validation set. Then, test on test period to see if the model will perform just as well. 

There is also another way to split training, validation and test sets with using **roll-forward partitioning.** 

<p align="center">
    <img src="img\roll-forward-partitioning.PNG" alt="roll-forward-partitioning" width=500>
</p>

## Metrics for Evaluating Performance



## References
* [Ungraded Lab - C4_W1_Lab_1_time_series](https://colab.research.google.com/drive/1_QdTh3jQxxAMCekxUbagkmL-mJKDUSom?usp=sharing)
* [Naïve Forecasting](https://www.avercast.in/blog/what-is-naive-forecasting-and-how-can-be-used-to-calculate-future-demand#:~:text=Na%C3%AFve%20forecasting%20is%20the%20technique,to%20the%20final%20observed%20value.)