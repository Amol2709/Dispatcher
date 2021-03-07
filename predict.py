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
import mysql.connector as sql
#warnings.filterwarnings("ignore", category=DeprecationWarning)

##################################################################################################################################################

import tensorflow_hub as hub



####################################################################################################################################################




class Predict:
	def __init__(self,status):
		self.status = status
		db_connection = sql.connect(host='localhost', database='my-db', user='root', password='root@123root@123',port=3307)
		self.df=pd.read_sql('SELECT * FROM finalmaindb', con=db_connection)
		
	
		print("*"*100)
		#print('Data CLeaning Started')
		#self.df['short_desc'] = self.df['short_desc'].astype(str)
		#self.df['long_desc'] = self.df['long_desc'].astype(str)
		#self.df['desc'] = self.df['short_desc'] + self.df['long_desc']
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

		self.training_desc = list(self.df['desc'])

		

		self.clean_train_desc = list(self.df['desc'])
		
		print("*"*100)
		print('Data Preparing Finished')

	

	def ModelPredictionAG(self,seed_text):

	
		print('*'*50)
		print('Prediction on AG : .........................................')
		# vocab_size = 1500
		# embedding_dim = 32
		# max_length = 150
		# trunc_type='post'
		# oov_tok = "<OOV>"
		# cross_check=3
		#self.seed_text =  self.clean_train_desc[3]
		self.seed_text = seed_text
		# self.tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
		# self.tokenizer.fit_on_texts(self.clean_train_desc)

		# token_list = self.tokenizer.texts_to_sequences([self.seed_text])[0]
		# token_list = pad_sequences([token_list], maxlen=max_length, truncating=trunc_type)


		###############################################################################################################################################
		

		re_model = tf.keras.models.load_model('Assignmentgroup_model.h5',custom_objects={'KerasLayer': hub.KerasLayer})
		predicted = re_model.predict(self.seed_text)

		################################################################################################################################################
		

		#print("Model Prediction : {}".format(list(self.le.classes_)[np.argmax(predicted)]))
		#print("Original Tag:      {}".format(self.df.iloc[cross_check][self.status]))
		#print(predicted.shape)
		#print(type(predicted))
		percent=100*predicted[0][np.argmax(predicted)]
		#print("Model Prediction : {} with percent {}".format(list(self.le.classes_)[np.argmax(predicted)],percent))
		predicted_ag = list(self.le.classes_)[np.argmax(predicted)]

		return predicted_ag,percent


	def ModelPredictionTAG(self,seed_text):
		#self.cleaning()
		print('*'*50)
		print('Prediction on TAG : .........................................')
		# vocab_size = 1500
		# embedding_dim = 32
		# max_length = 150
		# trunc_type='post'
		# oov_tok = "<OOV>"
		# cross_check=3
		#self.seed_text = self.clean_train_desc[3]#self.clean_train_desc[cross_check]
		self.seed_text = seed_text
		# self.tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
		# self.tokenizer.fit_on_texts(self.clean_train_desc)

		# token_list = self.tokenizer.texts_to_sequences([self.seed_text])[0]
		# token_list = pad_sequences([token_list], maxlen=max_length, truncating=trunc_type)

		#########################################################################################################################################


		re_model = tf.keras.models.load_model('ML_TAGmodel.h5',custom_objects={'KerasLayer': hub.KerasLayer})
		predicted = re_model.predict(self.seed_text)


		########################################################################################################################################

		#print(predicted)
		#print(len(list(self.le.classes_)))
		#print(np.argmax(predicted))

		#print(list(self.le.classes_)[np.argmax(predicted)])
		percent=100*predicted[0][np.argmax(predicted)]

		#print("Model Prediction : {} with percent {}".format(list(self.le.classes_)[np.argmax(predicted)],percent))
		#print(predicted.shape)
		#print(predicted)
		#print("Original Tag:      {}".format(self.df.iloc[cross_check][self.status]))
		predicted_tag = list(self.le.classes_)[np.argmax(predicted)]
		return predicted_tag,percent













