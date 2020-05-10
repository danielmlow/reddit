#!/usr/bin/env python3

'''
Authors: Daniel M. Low
License: Apache 2.0
'''
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from xgboost.sklearn import XGBModel
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest

from sklearn.pipeline import Pipeline

normalization_both = (StandardScaler(), MinMaxScaler())
normalization_std = (StandardScaler(),)

def parameters_all_models(y, dim_reduction):
	n_classes = len(np.unique(y))

	if dim_reduction:
		k = (2,8)
		n_neighbors = (16,32,64)
		parameters = [
			{
				'clf__estimator': [SGDClassifier(early_stopping=True, max_iter=5000), ],
				# SVM if hinge loss / logreg if log loss
				'normalization': (normalization_both),
				'umap__n_components': k,
				'umap__n_neighbors': n_neighbors,
				'clf__estimator__penalty': ('l2', 'elasticnet', 'l1'),
				'clf__estimator__loss': ['hinge', 'log'],
			},
			{
				'clf__estimator': [SVC(probability=False)],
				'normalization': (normalization_both),
				'clf__estimator__C': (0.01, 0.1, 1, 10, 100),
				'clf__estimator__kernel': ('rbf',),
				'umap__n_components': k,
				'umap__n_neighbors': n_neighbors,
			},
			{
				'clf__estimator': [
					XGBModel(objective='multi:softmax', num_class=n_classes, max_features='auto', n_jobs=-1)],
				'normalization': normalization_std,
				'clf__estimator__n_estimators': (32, 128),
				'clf__estimator__max_depth': (32, 64, 128),
				'clf__estimator__learning_rate': (0.01, 0.1),
				'umap__n_components': k,
				'umap__n_neighbors': n_neighbors,
			},
			{
				'clf__estimator': [ExtraTreesClassifier(max_features='auto', n_jobs=-1)],
				'normalization': normalization_std,
				'clf__estimator__n_estimators': (32, 128),
				'clf__estimator__max_depth': (32, 64, 128),
				'umap__n_components': k,
				'umap__n_neighbors': n_neighbors,
			},
			{
				'clf__estimator': [MLPClassifier(early_stopping=True, max_iter=200)],
				'normalization': normalization_std,
				'clf__estimator__batch_size': (32, 128, 512),
				'clf__estimator__hidden_layer_sizes': [(64, 16), (16, 16)],
				'clf__estimator__activation': ['relu'],
				'clf__estimator__alpha': [0.0001, 0.05],
				'clf__estimator__solver': ['adam'],
				'umap__n_components': k,
				'umap__n_neighbors': n_neighbors,
			},

		]

	else:

		k = (32, 64, 128,'all')



		parameters = [
			{
				'clf__estimator': [SGDClassifier(early_stopping=True, max_iter=5000),], # SVM if hinge loss / logreg if log loss
				'normalization': (normalization_both),
				'feature_selection__k': k,
				'clf__estimator__penalty': ('l2', 'elasticnet', 'l1'),
				'clf__estimator__loss': ['hinge','log'],
			},
			{
				'clf__estimator': [SVC(probability=False)],
				'normalization': (normalization_both),
				'clf__estimator__C': (0.01,0.1,1, 10,100),
				'clf__estimator__kernel': ('rbf',),
				'feature_selection__k': k,
			},
			{
				'clf__estimator': [XGBModel(objective='multi:softmax', num_class=n_classes, max_features='auto', n_jobs=-1)],
				'normalization': normalization_std,
				'clf__estimator__n_estimators': (32, 128),
				'clf__estimator__max_depth': (32, 64, 128),
				'clf__estimator__learning_rate': (0.01, 0.1),
				'feature_selection__k': k,
			},
			{
				'clf__estimator': [ExtraTreesClassifier(max_features='auto', n_jobs=-1)],
				'normalization': normalization_std,
				'clf__estimator__n_estimators': (32,128),
				'clf__estimator__max_depth':(32, 64, 128),
				'feature_selection__k': k,
			},
			{
			'clf__estimator': [MLPClassifier(early_stopping=True, max_iter=200)],
			'normalization': normalization_std,
			'clf__estimator__batch_size': (32,128,512),
			'clf__estimator__hidden_layer_sizes': [(256,32), (64, 32)],
			'clf__estimator__activation': ['relu'],
			'clf__estimator__alpha': [0.0001, 0.05],
			'clf__estimator__solver': ['adam'],
			'feature_selection__k': k,
			},

		]
	return parameters


def parameters_all_models_final(y, dim_reduction):
	if dim_reduction:
		k=2
	else:
		k = 'all'
	n_classes = len(np.unique(y))
	parameters = [
		{
			# SGD, train: 0.8278
			'clf__estimator': SGDClassifier(early_stopping=True, max_iter=5000), # SVM if hinge loss / logreg if log loss
			'normalization': MinMaxScaler(),
			'feature_selection__k': 'all',
			'clf__estimator__penalty': 'l1',
			'clf__estimator__loss': 'log',
		},
		{
			'clf__estimator': [SVC(kernel='rbf', probability=False)],
			'normalization': (normalization_both),
			'clf__estimator__C': (0.01,0.1,1, 10,100),
			'clf__estimator__gamma': ('scale','auto'),
			'feature_selection__k': k,
		},
		{
			'clf__estimator': [ExtraTreesClassifier()],
			'normalization': normalization_std,
			'clf__estimator__n_estimators': (16,32,128),
			'clf__estimator__max_depth':(32, 64, None),
			'feature_selection__k': k,
		},
		# default params: https://stackoverflow.com/questions/34674797/xgboost-xgbclassifier-defaults-in-python
		{
			# 'clf__estimator': [XGBModel(objective='multi:softmax',num_class=n_classes, max_features='auto')],
			'clf__estimator': [XGBModel()],
			'normalization': normalization_std,
			'clf__estimator__n_estimators': (16,32,128),
			'clf__estimator__max_depth':(32, 64),
			'clf__estimator__learning_rate': (0.01, 0.1, 0.3),
			'feature_selection__k': k,
		},
		{
		'clf__estimator': [MLPClassifier()],
		'normalization': normalization_std,
		'clf__estimator__batch_size': (512),
		'clf__estimator__hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
		'clf__estimator__activation': ['relu'],
		'clf__estimator__alpha': [0.0001, 0.05],
		'clf__estimator__solver': ['adam'],
		'feature_selection__k': k,
		},

	]
	return parameters


def final_pipeline(run_modelN):
	if run_modelN == 0:
		k = 'all'
		scaler = MinMaxScaler()
		clf = SGDClassifier(early_stopping=True, max_iter=5000, penalty='l1', loss='log')

	if run_modelN == 1:
		k = 'all'
		scaler = MinMaxScaler()
		clf = SGDClassifier(early_stopping=True, max_iter=5000, penalty='elasticnet', loss='log')

	elif run_modelN == 2:
		k = 'all'
		scaler = MinMaxScaler()
		clf = SVC(kernel='linear', probability=True)

	#
	# elif run_modelN == 3:
	# 	k = 'all'
	# 	scaler = MinMaxScaler()
	# 	clf = SVC(kernel='rbf', probability=True)

	elif run_modelN == 3:
		k = 'all'
		scaler = MinMaxScaler()
		clf = ExtraTreesClassifier(n_jobs=-1)

	elif run_modelN == 4:
		k = 'all'
		scaler = MinMaxScaler()
		clf = XGBModel(n_jobs=-1)

	# elif run_modelN == 6:
	# 	k = 'all'
	# 	scaler = MinMaxScaler()
	# 	clf = MLPClassifier(hidden_layer_sizes=(256),early_stopping=True,max_iter=1000,)

	pipeline = Pipeline([
		('normalization',scaler),
		('feature_selection', SelectKBest(k=k)),
		('clf', clf),
	])
	return pipeline