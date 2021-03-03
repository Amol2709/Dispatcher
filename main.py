import tensorflow as tf
import numpy as np
import pandas as pd
from TrainModelScratch import TrainModelScratch
from ReTraining import ReTraining
from Decide import Decide





def checkForAG():
	print("*"*100)
	print('Checking For Assignment Group')
	print("*"*100)
	obj = Decide('assignment_group')
	x,DF= obj.decideModel()


	print('Loading Previous Model')
	print("*"*100)
	re_model = tf.keras.models.load_model('Assignmentgroup_model.h5')
	print('Previous Model Detail....')
	re_model.summary()
	output_neurons=re_model.layers[-1].output_shape[1]
	print('Assignment_group present in ML Model: {}'.format(output_neurons))
	print('Assignment_group needed {}'.format(x))

	if output_neurons==x:
		#No need to retrain model from scratch
		print("*"*100)
		print('RETRAINING  - -- -> - - - - - -> - - - - - - - >OLD MODEL since No new Tag Added')
		ag_1 = ReTraining(DF,'assignment_group')
		ag_1.cleaning()
		ag_1.LoadAGModel('Assignmentgroup_model.h5')
		ag_1.ModelTraining(name='Assignmentgroup_model',epoch=2)
		
	else:
		print("*"*100)
		print('Need To Retrain Model From Scratch since Tag Increased.....')
		ag = TrainModelScratch(DF,'assignment_group')
		ag.cleaning()
		ag.AGModelBuilding()
		ag.ModelTraining(name='Assignmentgroup_model',epoch=2)
		# Retrain from scratch


def checkForTag():
	print("*"*100)
	print('RETRAINING  - -- -> - - - - - -> - - - - - - - >OLD MODEL since No new Tag Added')
	obj = Decide('cleaned_tags')
	x,DF= obj.decideModel()
	re_model = tf.keras.models.load_model('ML_TAGmodel.h5')
	print('Previous Model Detail....')
	re_model.summary()
	output_neurons=re_model.layers[-1].output_shape[1]
	print('ML Tag present in ML Model: {}'.format(output_neurons))
	print('ML Tag needed {}'.format(x))

	if output_neurons==x:
		#No need to retrain model from scratch
		print("*"*100)
		print('RETRAINING  - -- -> - - - - - -> - - - - - - - >OLD MODEL since No new Tag Added')
		tag_1 = ReTraining(DF,'cleaned_tags')
		tag_1.cleaning()
		tag_1.LoadAGModel('ML_TAGmodel.h5')
		tag_1.ModelTraining(name='ML_TAGmodel',epoch=2)
		
	else:
		print("*"*100)
		print('Need To Retrain Model From Scratch since Tag Increased.....')
		tag = TrainModelScratch(DF,'cleaned_tags')
		tag.cleaning()
		tag.TAGModelBuilding()
		tag.ModelTraining(name='ML_TAGmodel',epoch=2)
		# Retrain from scratch


checkForAG()
checkForTag()

