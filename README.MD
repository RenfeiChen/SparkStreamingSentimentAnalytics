# Twitter Sentiment Analytics using Apache Spark Streaming APIs and Python within a trained NaiveBayes model based on 160 millions data

In this project, I learnt about processing live data streams using Spark’s streaming APIs and Python. I trained a NaiveBayes model with the labeled 160 millions tweets data and use the model the get the sentiment analysis of real-time tweets. 

### Project Setup

#### Installing Required Python Libraries imported in the codes

##### Change the port in the TweetRead.py in order not to conflict with other process

##### Change the topic in the TweetRead.py file

#### Using the Twitter Streaming API In order to download the tweets from twitter streaming with script TweetRead.py. The script will need your twitter authentication tokens (keys).

Once you have your authentication tokens, create or update the twitter-app-credentials.txt with these credentials.

After updating the text file with your twitter keys, you can start downloading tweets from the twitter stream API

Note: This program should be kept running for collecting tweets.

##### Running the Stream Analysis Program: classify.py or you can set the size of each tweets data depends on your computational ability

##### Get the pie plot of positive and negative tweets in the real time 