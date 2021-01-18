"""
First part of the PN-AMFBR project

In this part, the question is that if adding new samples would increase accuracy
of prediction on real samples
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

class FirstAnalysis:

	def __init__(self, file_name, random_state = None, n_test = 20):

		self.file_name = file_name
		self.random_state = random_state
		self.n_test = n_test

		self.root_dir = "Data/"

		self._load_data()

	def _load_data(self):

		# Loading the csv file
		df = pd.read_csv(self.root_dir + self.file_name + ".csv", index_col = 0)

		# Slicing the test set from real samples
		real_df = df[df['Label'] == 'r']
		synth_df = df[df['Label'] == 's']

		test_df = real_df.sample(n = self.n_test, random_state = self.random_state)
		train_df_real = real_df.loc[set(real_df.index) - set(test_df.index)]

		# Shuffling the synth train set
		synth_df = synth_df.sample(frac = 1, random_state = self.random_state)

		# Dropping the RS column
		self.test_df = test_df.drop(columns = "Label", inplace = False)
		self.train_df_real = train_df_real.drop(columns = "Label", inplace = False)
		self.synth_df = synth_df.drop(columns = "Label", inplace = False)

		print ("Data is loaded...")

	def _get_regression_results(self, y_true, y_pred):
		return np.corrcoef(y_true, y_pred)[0][1]

	def run(self, N = 5):

		lin_errs, rf_errs = [], []
		percentages = np.linspace(0, 1, N)

		X_test, Y_test = self.test_df.iloc[:, :-1], self.test_df.iloc[:, -1]

		for perc in percentages:
			print (perc, 'percentage is about to analyzed in')

			# Getting the train set X and Y
			idx = int(perc * len(self.synth_df))
			temp_df = pd.concat([self.train_df_real, self.synth_df.iloc[:idx, :]])
			X_train, Y_train =  temp_df.iloc[:, :-1], temp_df.iloc[:, -1]

			# Training linear model
			lin_model = LinearRegression(fit_intercept = True, normalize=False)
			lin_model.fit(X_train, Y_train)

			# Getting the linear regression erros
			lin_test_err = self._get_regression_results(Y_test, lin_model.predict(X_test))
			lin_errs.append(lin_test_err)

			# Training random forest model
			rf_model = RandomForestRegressor(
							n_estimators = 20, 
							max_depth = 5,
							min_samples_split = 2,
							min_samples_leaf = 1,
							max_features = 'auto',
							random_state = self.random_state,
							n_jobs = -1,
							verbose = 1
							)
			rf_model.fit(X_train, Y_train)

			# Getting the random forest regression erros
			rf_test_err = self._get_regression_results(Y_test, rf_model.predict(X_test))
			rf_errs.append(rf_test_err)

		df = pd.DataFrame()
		df['Linear'] = lin_errs
		df['RF'] = rf_errs
		# df['rn_state'] = [self.random_state for _ in percentages]
		df.to_csv(f"Task1/Task1-{self.file_name}.csv")


		plt.plot(percentages, lin_errs, label = 'Linear')
		plt.plot(percentages, rf_errs, label = 'RandomForest')
		plt.xlabel("Percentage of train set")
		plt.ylabel("Correlation Coefficient")
		plt.title(f"{self.file_name}-{self.random_state}")
		plt.legend()
		plt.grid()
		plt.savefig(f"Task1/Task1-{self.file_name}.png")
		plt.show()

if __name__ == "__main__":

	rn_state = int(np.random.random()*1000)
	rn_state = 289

	print (f"\n\nrn_state:{rn_state}\n\n")

	myAnaysis = FirstAnalysis(file_name = 'ammonium22',
						random_state = rn_state,
						n_test = 20)
	myAnaysis.run(N = 11)





		
