from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import socket
import json
import csv
import re
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
from collections import namedtuple

# Set up your credentials
consumer_key=''
consumer_secret=''
access_token =''
access_secret=''

class TweetsListener(StreamListener):
  count = 0
  def __init__(self, csocket):
      self.client_socket = csocket

  def on_data(self, data):
      try:
          msg = json.loads( data )
          # 100 is the size of each csv
          file = 'Data' + str(TweetsListener.count // 100) + '.csv'
          with open(file, 'a', encoding='utf-8-sig', newline='') as f:
              writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
              text = msg["text"].strip()
              index = text.find(":")
              if (index != -1):
                text = text[index + 1:]
              text = text.replace("\n", " ").strip()
              text = re.sub(r"http\S+", "", text)
              data = [text]
              if text.startswith("/") == False:
                TweetsListener.count += 1
                print(text)
                writer.writerow(data)
          self.client_socket.send( msg['text'].encode('utf-8') )
          return True
      except BaseException as e:
          print("Error on_data: %s" % str(e))
      return True

  def on_error(self, status):
      print(status)
      return True

def sendData(c_socket):
  auth = OAuthHandler(consumer_key, consumer_secret)
  auth.set_access_token(access_token, access_secret)
  twitter_stream = Stream(auth, TweetsListener(c_socket))
  # life is the target topic, change it if you want
  twitter_stream.filter(languages=["en"], track=['life'])

if __name__ == "__main__":
  host = "127.0.0.1"
  port = 9961
  count = 0
  sc = SparkContext()
  ssc = StreamingContext(sc, 10)
  sqlContext = SQLContext(sc)
  socket_stream = ssc.socketTextStream(host, port)
  lines = socket_stream.window(20)
  fields = ("tag", "count")
  Tweet = namedtuple('Tweet', fields)

  (lines.flatMap(lambda text: text.split(" "))  # Splits to a list
   .filter(lambda word: word.lower().startswith("#"))  # Checks for hashtag calls
   .map(lambda word: (word.lower(), 1))  # Lower cases the word
   .reduceByKey(lambda a, b: a + b)  # Reduces
   .map(lambda rec: Tweet(rec[0], rec[1]))  # Stores in a Tweet Object
   .foreachRDD(lambda rdd: rdd.toDF().limit(10).registerTempTable("tweets")))

  ssc.start()
  s = socket.socket()         # Create a socket object
  s.bind((host, port))        # Bind to the port
  print("Listening on port: %s" % str(port))
  s.listen(5)                 # Now wait for client connection.
  c = s.accept()        # Establish connection with client.
  sendData( c )

