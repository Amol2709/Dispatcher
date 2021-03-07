# import warnings

# warnings.filterwarnings('ignore')
 
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
from CustomCallBack import MyCallback



###############################################################################################################
import tensorflow_hub as hub
from sklearn.utils import class_weight
import matplotlib.pyplot as plt



###############################################################################################################

class TrainModelScratch:
	def __init__(self,df,status):

		self.df = df
		self.status = status
		
	
		print("*"*100)
		print('Data Preparing Started............')
		
	
		self.df = self.df[["desc",self.status]]
		
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

		#self.training_desc = list(self.df['desc'])

		
		self.clean_train_desc = list(self.df['desc'])



		#########################################################################################################################################################

		self.DICT ={}
		for i in range(0,len(self.x)):
			self.DICT[list(self.le.classes_)[i]] = list(self.Labels)[i]
		y_helper = []
		T=list(self.df[self.status])
		for i in range(0,len(T)):
			y_helper.append(self.DICT[T[i]])


		class_weights = class_weight.compute_class_weight('balanced',self.Labels,y_helper)
		self.class_weights = dict(enumerate(class_weights))


		##################################################################################################################################################
		
		print('Data Preparing Finished.....')

	def AGModelBuilding(self):
		print("*"*100)
		print('Building New Model..............')


		###########################################################################################################################
		self.hub_layer = hub.KerasLayer("model",input_shape=[], dtype=tf.string) #......remember to give path like app/model/....(just like we did in main.py for h5 files)
		self.model = tf.keras.Sequential()

		self.model.add(self.hub_layer)
		# self.model.add(tf.keras.layers.Dense(1024, activation='relu'))
		# self.model.add(tf.keras.layers.Dense(512, activation='relu'))
		# self.model.add(tf.keras.layers.Dense(264, activation='relu'))
		# self.model.add(tf.keras.layers.Dense(1024, activation='relu'))
		# self.model.add(tf.keras.layers.Dense(512, activation='relu'))
		# self.model.add(tf.keras.layers.Dense(264, activation='relu'))




		#self.model.add(tf.keras.layers.Dense(128, activation='relu'))
		#self.model.add(tf.keras.layers.Dropout(0.3))
		self.model.add(tf.keras.layers.Dense(128, activation='relu'))
		self.model.add(tf.keras.layers.Dense(512, activation='relu'))
		# self.model.add(tf.keras.layers.Dense(1024, activation='relu'))
		# self.model.add(tf.keras.layers.Dense(512, activation='relu'))
		#self.model.add(tf.keras.layers.Dense(8*len(self.Labels), activation='relu'))
		#self.model.add(tf.keras.layers.Dense(4*len(self.Labels), activation='relu'))
		self.model.add(tf.keras.layers.Dense(2*len(self.Labels), activation='relu'))
		
		#self.model.add(tf.keras.layers.Dropout(0.1))
		self.model.add(tf.keras.layers.Dense(len(self.Labels), activation='softmax'))
		self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
		self.model.summary()









		#############################################################################################################################









		# vocab_size = 1500
		# embedding_dim = 32
		# max_length = 150
		# trunc_type='post'
		# oov_tok = "<OOV>"
		# tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
		# tokenizer.fit_on_texts(self.clean_train_desc)

		# word_index = tokenizer.word_index
		# clean_train_sequences = tokenizer.texts_to_sequences(self.clean_train_desc)
		# self.clean_train_padded = pad_sequences(clean_train_sequences,maxlen=max_length, truncating=trunc_type)


		# self.model = tf.keras.Sequential([
		# 	tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
		# 	#tf.keras.layers.Flatten(),
		# 	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
		# 	tf.keras.layers.Dropout(0.2),
		# 	tf.keras.layers.Dense(2*len(self.Labels), activation='relu'),
		# 	tf.keras.layers.Dense(len(self.Labels), activation='softmax')
		# 	])
		# self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
		# self.model.summary()




	def TAGModelBuilding(self):
		print("*"*100)
		print('Building New Model..............')


		#####################################################################################################################

		self.hub_layer = hub.KerasLayer("model",input_shape=[], dtype=tf.string) #......remember to give path like app/model/....(just like we did in main.py for h5 files)
		self.model = tf.keras.Sequential()

		self.model.add(self.hub_layer)
		# self.model.add(tf.keras.layers.Dense(1024, activation='relu'))
		# self.model.add(tf.keras.layers.Dense(1024, activation='relu'))
		# self.model.add(tf.keras.layers.Dense(512, activation='relu'))
		
		# self.model.add(tf.keras.layers.Dense(512, activation='relu'))
		




		# self.model.add(tf.keras.layers.Dense(128, activation='relu'))
		#self.model.add(tf.keras.layers.Dropout(0.3))
		self.model.add(tf.keras.layers.Dense(128, activation='relu'))
		self.model.add(tf.keras.layers.Dense(512, activation='relu'))
		# self.model.add(tf.keras.layers.Dense(1024, activation='relu'))
		# self.model.add(tf.keras.layers.Dense(512, activation='relu'))
		self.model.add(tf.keras.layers.Dense(2*len(self.Labels), activation='relu'))
		#self.model.add(tf.keras.layers.Dropout(0.1))
		self.model.add(tf.keras.layers.Dense(len(self.Labels), activation='softmax'))
		self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
		self.model.summary()
		#####################################################################################################################


		# vocab_size = 1500
		# embedding_dim = 32
		# max_length = 150
		# trunc_type='post'
		# oov_tok = "<OOV>"
		# tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
		# tokenizer.fit_on_texts(self.clean_train_desc)

		# word_index = tokenizer.word_index
		# clean_train_sequences = tokenizer.texts_to_sequences(self.clean_train_desc)
		# self.clean_train_padded = pad_sequences(clean_train_sequences,maxlen=max_length, truncating=trunc_type)


		# self.model = tf.keras.Sequential([
		# 	tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
		# 	tf.keras.layers.Flatten(),
		# 	#tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
		# 	tf.keras.layers.Dropout(0.2),
		# 	tf.keras.layers.Dense(2*len(self.Labels), activation='relu'),
		# 	tf.keras.layers.Dense(len(self.Labels), activation='softmax')
		# 	])
		# self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
		# self.model.summary()


	def ModelTraining(self,name,epoch=1):
		print("*"*100)
		print('Training of New Model Started............')
		self.num_epochs = epoch
		self.name=name
		callbacks = MyCallback()
		#############################################################################################################################


		self.history = self.model.fit(np.array(self.clean_train_desc), self.train_label, epochs=self.num_epochs,callbacks=[callbacks],class_weight=self.class_weights,shuffle=True)
		scores = self.model.evaluate(np.array(self.clean_train_desc),self.train_label)

		##############################################################################################################
		plt.plot(self.history.history['accuracy'])
		plt.plot(self.history.history['loss'])
		plt.title('model Detail: {}'.format(self.status))
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['accuracy', 'loss'], loc='upper left')
		plt.savefig('model Detail_{}.jpg'.format(self.status))

		###############################################################################################################

		
		#############################################################################################################################
		print("Accuracy: %.2f%%" % (scores[1]*100))
		#############################################################################################################################################################
		# just make sure this................................
		self.model.save(self.name+'.h5')
		########################################################################################################################################################
		print(self.name)
		print('model save succesfully')













