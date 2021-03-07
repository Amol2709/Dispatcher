
# import warnings

# warnings.filterwarnings('ignore')
 



import tensorflow as tf
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from sklearn import preprocessing
from tqdm import tqdm
import mysql.connector as sql
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#warnings.filterwarnings("ignore", category=DeprecationWarning)
###########################################################################################################################################
#change
import tensorflow_hub as hub



##########################################################################################################################################

class AgAccuracy:
	def __init__(self,status):
		self.status = status
		db_connection = sql.connect(host='localhost', database='my-db', user='root', password='root@123root@123',port=3307)
		self.df=pd.read_sql('SELECT * FROM finalmaindb', con=db_connection)
		
		print("*"*100)
		#print('Data Preparing Started')
		
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
		

		self.train_label = np.zeros((self.df.count()[0],len(self.Labels)))
		#print(list(self.le.classes_))
		for i in range(0,self.df.count()[0]):
			#print(self.df[self.status][i])
			Index=list(self.le.classes_).index(self.df[self.status][i])
			self.train_label[i,:] = b[Index,:]

		#self.training_desc = list(self.df['desc'])
		self.clean_train_desc = list(self.df['desc'])
		
		print("*"*100)
		print('Data Preparing Finished')
		print("*"*100)
		print('Loading Old Assignnment Group Model From Disk For ReTraining ............')

		print("*"*100)
		#print('Building New Model..............')
		





		#self.trained_model = trained_model
		#########################################################################################################################################################
		#change

		self.model = tf.keras.models.load_model('Assignmentgroup_model.h5',custom_objects={'KerasLayer': hub.KerasLayer})
		self.model.summary()

		#############################################################################################################################################################




	# def LoadTAGModel(self):
	# 	print("*"*100)
	# 	print('Loading Old ML Tag Model From Disk For ReTraining ............')
	# 	#self.trained_model = trained_model
	# 	self.model = tf.keras.models.load_model('ML_TAGmodel.h5')
	# 	self.model.summary()


	def ModelAcc(self):
		print("*"*100)
		print('ReTraining Started............')
		#self.num_epochs = epoch
		#self.name=name
		
		# vocab_size = 1500
		# embedding_dim = 32
		# max_length = 150
		# trunc_type='post'
		# oov_tok = "<OOV>"
		# self.tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
		# self.tokenizer.fit_on_texts(self.clean_train_desc)

		# word_index = self.tokenizer.word_index
		# clean_train_sequences = self.tokenizer.texts_to_sequences(self.clean_train_desc)
		# self.clean_train_padded = pad_sequences(clean_train_sequences,maxlen=max_length, truncating=trunc_type)

		#history = self.model.fit(self.clean_train_padded, self.train_label, epochs=self.num_epochs)
		#########################################################################################################################################################################

		#change
		scores = self.model.evaluate(np.array(self.clean_train_desc),self.train_label)


		#####################################################################################################################################################
		print('----------SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS-----------------------------------------')
		print(scores)
		#print("Accuracy: %.2f%%" % (scores[1]*100))
		#self.model.save(self.name+'.h5')
		#print('model save succesfully')
		return scores[1]*100


	