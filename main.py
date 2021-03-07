# import warnings

# warnings.filterwarnings('ignore')
 
import tensorflow as tf
import numpy as np
import pandas as pd
from TrainModelScratch import TrainModelScratch
from ReTraining import ReTraining
from Decide import Decide
from predict import Predict
from datetime import date

from AGAccuracy import AgAccuracy
from TAGAccuracy import TagAccuracy
# import warnings

# warnings.filterwarnings('ignore')
 


##################################################################################################################################################

import tensorflow_hub as hub


####################################################################################################################################################



############# Only Assumption is database is upto date and neat and clean-------------------------
class reTrain:
	def __init__(self):
		self.epoch = 5
		pass
	def checkForAG(self):
		
		print("*"*100)
		print('Checking For Assignment Group')
		print("*"*100)
		obj = Decide('assignment_group')
		x,self.DF= obj.decideModel()


		print('Loading Previous Model')
		print("*"*100)
		###########################################################################################################################################


		re_model = tf.keras.models.load_model('Assignmentgroup_model.h5',custom_objects={'KerasLayer': hub.KerasLayer})


		##############################################################################################################################################
		print('Previous Model Detail....')
		re_model.summary()
		output_neurons=re_model.layers[-1].output_shape[1]
		print('Assignment_group present in ML Model: {}'.format(output_neurons))
		print('Assignment_group needed {}'.format(x))

		if output_neurons==x:
			#No need to retrain model from scratch
			print("*"*100)
			print('RETRAINING  - -- -> - - - - - -> - - - - - - - >OLD MODEL since No new Tag Added')
			ag_1 = ReTraining(self.DF,'assignment_group')
			#ag_1.cleaning()
			ag_1.LoadAGModel('Assignmentgroup_model.h5')
			ag_1.ModelTraining(name='Assignmentgroup_model',epoch=self.epoch)
			#ag_1.ModelPredictionAG()
			
		else:
			print("*"*100)
			print('Need To Retrain Model From Scratch since Tag Increased.....')
			ag = TrainModelScratch(self.DF,'assignment_group')
			#ag.cleaning()
			ag.AGModelBuilding()
			ag.ModelTraining(name='Assignmentgroup_model',epoch=self.epoch)
			# Retrain from scratch
	def checkForTag(self):
		print("*"*100)
		print('RETRAINING  - -- -> - - - - - -> - - - - - - - >OLD MODEL since No new Tag Added')
		obj_2 = Decide('cleaned_tags')
		x,self.DF_2= obj_2.decideModel()
		##############################################################################################################################################
		re_model = tf.keras.models.load_model('ML_TAGmodel.h5',custom_objects={'KerasLayer': hub.KerasLayer})
		######################################################################################################################################
		print('Previous Model Detail....')
		re_model.summary()
		output_neurons=re_model.layers[-1].output_shape[1]
		print('ML Tag present in ML Model: {}'.format(output_neurons))
		print('ML Tag needed {}'.format(x))
		if output_neurons==x:
			#No need to retrain model from scratch
			print("*"*100)
			print('RETRAINING  - -- -> - - - - - -> - - - - - - - >OLD MODEL since No new Tag Added')
			tag_1 = ReTraining(self.DF_2,'cleaned_tags')
			#tag_1.cleaning()
			tag_1.LoadTAGModel('ML_TAGmodel.h5')
			tag_1.ModelTraining(name='ML_TAGmodel',epoch=self.epoch)
			#tag_1.ModelPredictionTAG()
		else:
			print("*"*100)
			print('Need To Retrain Model From Scratch since Tag Increased.....')
			tag = TrainModelScratch(self.DF_2,'cleaned_tags')
			#tag.cleaning()
			tag.TAGModelBuilding()
			tag.ModelTraining(name='ML_TAGmodel',epoch=self.epoch)
			# Retrain from scratch



#-----------------------------------
# main_obj = reTrain()
# main_obj.checkForAG()
# main_obj.checkForTag()
#---------------------------------------
pred_obj_1 = Predict('assignment_group')
pred_obj_2 = Predict('cleaned_tags')



sent = ['uwf ctp failed jobs alert transformjobhello app ops team details failed catapult jobs last minutes job type transformjob please take high priority ctpjobid external tracking data process start time last error multistep reported multistep reported vantage reported message reported child ctpjobid parent multistepjob message reported child ctpjobid parent multistepjob accurate ctpjobid external tracking data process start time last error multistep reported multistep reported vantage reported source conform error invalid duration message reported child ctpjobid parent multistepjob message reported child ctpjobid parent multistepjob accurate ctpjobid external tracking data process start time last error multistep reported error post processing unable move output file expected location multistep job completed output file not exist wip directory message reported child ctpjobid parent multistepjob accurate ctpjobid external tracking data na process start time last error multistep reported vantage reported message reported child ctpjobid parent multistepjob accurate ctpjobid external tracking data na process start time last error vantage reported accurate ctpjobid external tracking data na process start time last error error post processing unable move output file expected location multistep job completed output file not exist wip directory accurate ctpjobid external tracking data na process start time last error multistep reported vantage reported source conform error invalid duration message reported child ctpjobid parent multistepjob accurate ctpjobid external tracking data na process start time last error vantage reported source conform error invalid duration accurate note auto generated email please not reply mailbox not monitored']

import re
from bs4 import BeautifulSoup
from tqdm import tqdm
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
def decontracted(phrase):
	phrase = re.sub(r"won't", "will not", phrase)
	phrase = re.sub(r"can\'t", "can not", phrase)
	phrase = re.sub(r"n\'t", " not", phrase)
	phrase = re.sub(r"\'re", " are", phrase)
	phrase = re.sub(r"\'s", " is", phrase)
	phrase = re.sub(r"\'d", " would", phrase)
	phrase = re.sub(r"\'ll", " will", phrase)
	phrase = re.sub(r"\'t", " not", phrase)
	phrase = re.sub(r"\'ve", " have", phrase)
	phrase = re.sub(r"\'m", " am", phrase)
	return phrase
desc = []
count = 0
# tqdm is for printing the status bar
for sentance in tqdm(sent):
	try:
		sentance = re.sub(r"http\S+", "", sentance)
		sentance = BeautifulSoup(sentance, "html.parser").get_text()
		sentance = decontracted(sentance)
		sentance = re.sub("\S*\d\S*", "", sentance).strip()
		sentance = re.sub('[^A-Za-z]+', ' ', sentance)
		# https://gist.github.com/sebleier/554280
		sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
		desc.append(sentance.strip())
	except:
		print('Exception Ocuured During Cleaning of data')
	else:
		count= count+1

sent = desc.copy()


pred_ag,per_ag=pred_obj_1.ModelPredictionAG(sent)
pred_tag,per_tag=pred_obj_2.ModelPredictionTAG(sent)

print("Predicted Group : {} with percentage {}".format(pred_ag,per_ag))
print("Predicted Tag: {} with percentage {}".format(pred_tag,per_tag))

# #-----------------------------------------------------------------------------

# Acc_ag = AgAccuracy('assignment_group')
# Acc_tag = TagAccuracy('cleaned_tags')


# print('Accuracy of assignment_group: {}'.format(Acc_ag.ModelAcc()))
# print('Accuracy of Tag: {}'.format(Acc_tag.ModelAcc()))

