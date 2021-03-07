# import warnings

# warnings.filterwarnings('ignore')
 
import numpy as np
import pandas as pd
import mysql.connector as sql



class Decide:
	def __init__(self,status):
		print('*'*100)
		print('Establishing Connection with Database and fecthin data..')
		db_connection = sql.connect(host='localhost', database='my-db', user='root', password='root@123root@123',port=3307)
		self.df=pd.read_sql('SELECT * FROM finalmaindb', con=db_connection)
		#self.df = pd.read_excel('viacomdatacleanup.xlsx', engine='openpyxl',sheet_name='Sheet1')
		#self.DF= self.df.copy()
		#self.length= self.df.count()[0]
		self.status = status
		# if self.status == 'cleaned_tags':
		# 	self.df=self.df.dropna(how='any')
		# else:
		# 	pass


	def decideModel(self):

		print('-----------------------INSIDE CLEANINNG----------------')
		#self.df['short_desc'] = self.df['short_desc'].astype(str)
		#self.df['long_desc'] = self.df['long_desc'].astype(str)

		#self.df['desc'] = self.df['short_desc'] + self.df['long_desc']

		#self.df[self.status] = self.df[self.status].astype(str)

		#self.df = self.df[["desc",self.status]]
		# self.df=self.df.dropna(how='any')
		# self.df=self.df.loc[self.df[self.status]!='0']
		# self.df=self.df.loc[self.df[self.status]!='']
		# self.df=self.df.reset_index()
		# del self.df['index']
		self.A=dict(self.df[self.status].value_counts())
		#print(A)
		self.x=list(self.A.keys()) # number of new tags
		self.y= list(self.A.values())

		return (len(self.x),self.df)



		




