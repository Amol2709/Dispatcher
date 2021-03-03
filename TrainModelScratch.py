import tensorflow as tf
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from sklearn import preprocessing
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#warnings.filterwarnings("ignore", category=DeprecationWarning)


class TrainModelScratch:
	def __init__(self,df,status):
		self.df = df
		self.status = status
		if self.status == 'cleaned_tags':
			self.df=self.df.dropna(how='any')
		else:
			pass
	def decontracted(self,phrase):
		self.phrase = phrase
		self.phrase = re.sub(r"won't", "will not", self.phrase)
		self.phrase = re.sub(r"can\'t", "can not", self.phrase)
		self.phrase = re.sub(r"n\'t", " not", self.phrase)
		self.phrase = re.sub(r"\'re", " are", self.phrase)
		self.phrase = re.sub(r"\'s", " is", self.phrase)
		self.phrase = re.sub(r"\'d", " would", self.phrase)
		self.phrase = re.sub(r"\'ll", " will", self.phrase)
		self.phrase = re.sub(r"\'t", " not", self.phrase)
		self.phrase = re.sub(r"\'ve", " have", self.phrase)
		self.phrase = re.sub(r"\'m", " am", self.phrase)
		return self.phrase
	def cleaning(self):
		print("*"*100)
		print('Data Cleaning Started............')
		self.df['short_desc'] = self.df['short_desc'].astype(str)
		self.df['long_desc'] = self.df['long_desc'].astype(str)
		self.df['desc'] = self.df['short_desc'] + self.df['long_desc']
		self.df[self.status] = self.df[self.status].astype(str)
		self.df = self.df[["desc",self.status]]
		self.df=self.df.dropna(how='any')
		self.df=self.df.loc[self.df[self.status]!='0']
		self.df=self.df.reset_index()
		del self.df['index']
		self.A=dict(self.df[self.status].value_counts())
		self.x=list(self.A.keys()) # number of new tags
		self.y= list(self.A.values())



		self.le = preprocessing.LabelEncoder()
		self.le.fit(self.x)
		self.Labels=self.le.transform(list(self.le.classes_))
		a =self.Labels.copy()
		b = np.zeros((a.size, a.max()+1))
		b[np.arange(a.size),a] = 1
		#print(b.shape)
		#print(list(self.le.classes_))

		self.train_label = np.zeros((self.df.count()[0],len(self.Labels)))
		#print(list(self.le.classes_))
		for i in range(0,self.df.count()[0]):
			#print(self.df[self.status][i])
			Index=list(self.le.classes_).index(self.df[self.status][i])
			self.train_label[i,:] = b[Index,:]

		self.training_desc = list(self.df['desc'])

		stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])


		self.clean_train_desc = []
		count = 0
		for sentance in tqdm(self.df['desc'].values):
			try:
				sentance = re.sub(r"http\S+", "", sentance)
				sentance = BeautifulSoup(sentance, "html.parser").get_text()
				sentance = self.decontracted(sentance)
				sentance = re.sub("\S*\d\S*", "", sentance).strip()
				sentance = re.sub('[^A-Za-z]+', ' ', sentance)
				sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
				self.clean_train_desc.append(sentance.strip())
			except:
				print(sentance)
				print(count)
			else:
				count= count+1
		print('Data Cleaning Finished.....')

	def AGModelBuilding(self):
		print("*"*100)
		print('Building New Model..............')
		vocab_size = 1500
		embedding_dim = 32
		max_length = 150
		trunc_type='post'
		oov_tok = "<OOV>"
		tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
		tokenizer.fit_on_texts(self.clean_train_desc)

		word_index = tokenizer.word_index
		clean_train_sequences = tokenizer.texts_to_sequences(self.clean_train_desc)
		self.clean_train_padded = pad_sequences(clean_train_sequences,maxlen=max_length, truncating=trunc_type)


		self.model = tf.keras.Sequential([
			tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
			#tf.keras.layers.Flatten(),
			tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
			tf.keras.layers.Dropout(0.2),
			tf.keras.layers.Dense(2*len(self.Labels), activation='relu'),
			tf.keras.layers.Dense(len(self.Labels), activation='softmax')
			])
		self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
		self.model.summary()




	def TAGModelBuilding(self):
		print("*"*100)
		print('Building New Model..............')
		vocab_size = 1500
		embedding_dim = 32
		max_length = 150
		trunc_type='post'
		oov_tok = "<OOV>"
		tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
		tokenizer.fit_on_texts(self.clean_train_desc)

		word_index = tokenizer.word_index
		clean_train_sequences = tokenizer.texts_to_sequences(self.clean_train_desc)
		self.clean_train_padded = pad_sequences(clean_train_sequences,maxlen=max_length, truncating=trunc_type)


		self.model = tf.keras.Sequential([
			tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
			tf.keras.layers.Flatten(),
			#tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
			tf.keras.layers.Dropout(0.2),
			tf.keras.layers.Dense(2*len(self.Labels), activation='relu'),
			tf.keras.layers.Dense(len(self.Labels), activation='softmax')
			])
		self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
		self.model.summary()


	def ModelTraining(self,name,epoch=1):
		print("*"*100)
		print('Training of New Model Started............')
		self.num_epochs = epoch
		self.name=name
		history = self.model.fit(self.clean_train_padded, self.train_label, epochs=self.num_epochs)
		scores = self.model.evaluate(self.clean_train_padded,self.train_label)
		print("Accuracy: %.2f%%" % (scores[1]*100))
		self.model.save(self.name+'.h5')
		print('model save succesfully')











