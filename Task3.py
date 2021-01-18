"""
Third part of the PN-AMFBR project

In this part, we aim to get the feature importances in datasets using RF
"""

import os
import pandas as pd
import numpy as np
from collections import OrderedDict
import pprint
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.ensemble import RandomForestRegressor

class ThirdAnalysis:

	def __init__(self, file_name, random_state = None, n_test = 20, n_simulations = 5):

		self.file_name = file_name
		self.random_state = random_state
		self.n_test = n_test

		self.n_simulations = n_simulations

		self.root_dir = "Data/"

		df = pd.read_csv(self.root_dir + self.file_name + ".csv", index_col = 0)
		self.df = df.drop(columns = "Label", inplace = False)

	def _get_train_test(self):

		test_df = self.df.sample(n = self.n_test, random_state = self.random_state)
		train_df = self.df.loc[set(self.df.index) - set(test_df.index)]

		return train_df, test_df

	def _report_feature_importance(self,
		features_vals,
		features_names):

	    print ("About to conduct feature importance")
	    best_features = dict(zip(features_names, abs(features_vals)))
	    features_ = pd.Series(OrderedDict(sorted(best_features.items(), key=lambda t: t[1], reverse =True)))
	    
	    temp_df = pd.DataFrame()
	    temp_df['features'] = features_names
	    temp_df['realtive importance'] = abs(features_vals)
	    temp_df.to_csv("Task3/" + f"{self.file_name}-FS.csv")

	    plt.clf()
	    ax = features_.plot(kind='bar', title = self.file_name)
	    fig = ax.get_figure()
	    plt.tight_layout()
	    fig.savefig("Task3/" + f'{self.file_name}-FS.png')
	    del fig
	    plt.close()

	def run(self):

		# Holder of feature importance values
		f_imp = np.zeros(len(self.df.columns) - 1)

		# doing it several times
		for i in range (self.n_simulations):

			print (f"{i}/{self.n_simulations} of {self.file_name}")

			# Get the train and test sets
			train_df, test_df = self._get_train_test()
			X_train, Y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1] 

			# Training random forest model
			rf_model = RandomForestRegressor(
							n_estimators = 1000, 
							max_depth = None,
							min_samples_split = 2,
							min_samples_leaf = 1,
							max_features = 'auto',
							random_state = self.random_state,
							n_jobs = -1,
							verbose = 0
							)
			rf_model.fit(X_train, Y_train)

			f_imp += rf_model.feature_importances_

		# Getting the average
		f_imp = f_imp / self.n_simulations

		# Getting the results
		self._report_feature_importance(f_imp, X_train.columns)


def exec():

	for species in ['nitrite', 'nitrate', 'ammonium']:
		for n in [4]:
			file_name = f"{species}2{n}"
			
			myAnaysis = ThirdAnalysis(file_name = file_name,
				n_simulations = 100)
			myAnaysis.run()

if __name__ == "__main__":

	exec()





		
