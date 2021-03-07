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
from datetime import date
from datetime import datetime

from CustomCallBack import MyCallback

###################################################################################################################################
import tensorflow_hub as hub


#####################################################################################################################################3
#warnings.filterwarnings("ignore", category=DeprecationWarning)

class ReTraining:
	def __init__(self,df,status):
		self.df = df
		self.status = status
		print("*"*100)
		print('Data Preparing Started')
		
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

	def LoadAGModel(self,trained_model):
		print("*"*100)
		print('Loading Old Assignnment Group Model From Disk For ReTraining ............')

		print("*"*100)
		#print('Building New Model..............')
		





		self.trained_model = trained_model
		##########################################################################################################################################


		self.model = tf.keras.models.load_model(self.trained_model,custom_objects={'KerasLayer': hub.KerasLayer})


		#############################################################################################################################################
		self.model.summary()




	def LoadTAGModel(self,trained_model):
		print("*"*100)
		print('Loading Old ML Tag Model From Disk For ReTraining ............')
		self.trained_model = trained_model

		#############################################################################################################################################


		self.model = tf.keras.models.load_model(self.trained_model,custom_objects={'KerasLayer': hub.KerasLayer})

		##############################################################################################################################################
		self.model.summary()


	def ModelTraining(self,name,epoch=1):
		print("*"*100)
		print('ReTraining Started............')
		self.num_epochs = epoch
		self.name=name
		
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

		callbacks = MyCallback()
		######################################################################################################################################


		history = self.model.fit(np.array(self.clean_train_desc), self.train_label, epochs=self.num_epochs,callbacks=[callbacks])
		scores = self.model.evaluate(np.array(self.clean_train_desc),self.train_label)


		##########################################################################################################################################
		print("Accuracy: %.2f%%" % (scores[1]*100))
		############################################################################################################################################
		#...just make sure this
		self.model.save(self.name+'.h5')
		#############################################################################################################################################
		print('model save succesfully')


	# def ModelPredictionAG(self):
	# 	print('*'*50)
	# 	print('Prediction on AG : .........................................')
	# 	max_length = 150
	# 	trunc_type='post'
	# 	cross_check=3
	# 	self.seed_text = self.clean_train_desc[cross_check]
	# 	#self.seed_text = seed_text
	# 	token_list = self.tokenizer.texts_to_sequences([self.seed_text])[0]
	# 	token_list = pad_sequences([token_list], maxlen=max_length, truncating=trunc_type)
	# 	re_model = tf.keras.models.load_model('Assignmentgroup_model.h5')
	# 	predicted = re_model.predict(token_list,verbose=0)

	# 	print("Model Prediction : {}".format(list(self.le.classes_)[np.argmax(predicted)]))
	# 	print("Original Tag:      {}".format(self.df.iloc[cross_check][self.status]))


	# def ModelPredictionTAG(self):
	# 	print('*'*50)
	# 	print('Prediction on TAG : .........................................')
	# 	max_length = 150
	# 	trunc_type='post'
	# 	cross_check=3
	# 	self.seed_text = self.clean_train_desc[cross_check]
	# 	#self.seed_text = seed_text
	# 	token_list = self.tokenizer.texts_to_sequences([self.seed_text])[0]
	# 	token_list = pad_sequences([token_list], maxlen=max_length, truncating=trunc_type)
	# 	re_model = tf.keras.models.load_model('ML_TAGmodel.h5')
	# 	predicted = re_model.predict(token_list,verbose=0)

	# 	#print(predicted)
	# 	#print(len(list(self.le.classes_)))
	# 	#print(np.argmax(predicted))

	# 	#print(list(self.le.classes_)[np.argmax(predicted)])

	# 	print("Model Prediction : {}".format(list(self.le.classes_)[np.argmax(predicted)]))
	# 	print("Original Tag:      {}".format(self.df.iloc[cross_check][self.status]))














