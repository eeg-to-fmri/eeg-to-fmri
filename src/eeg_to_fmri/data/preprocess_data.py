from eeg_to_fmri.data import data_utils, eeg_utils, fmri_utils

import sys

import numpy as np

from sklearn.utils import shuffle

import tensorflow as tf

import gc

#should eeg_limit be true??
def dataset(dataset, n_individuals=8, interval_eeg=6, ind_volume_fit=True, raw_eeg=False, standardize_fmri=True, standardize_eeg=True, iqr=True, file_output=None, verbose=False):

	if(verbose):
		if(file_output == None):
			print("I: Starting to Load Data")	
		else:
			print("I: Starting to Load Data", file=file_output)

	#TR of fmri and window size of STFT
	f_resample=getattr(fmri_utils, "TR_"+dataset)

	if(dataset in ["02","03","04","05"]):
		eeg_limit=True
		eeg_f_limit=135
	else:
		eeg_limit=False
		eeg_f_limit=135

	eeg_train, fmri_train, scalers = data_utils.load_data(list(range(n_individuals)), raw_eeg=raw_eeg, n_voxels=None, 
															bold_shift=getattr(fmri_utils, "bold_shift_"+dataset), n_partitions=25, 
															mutate_bands=False,
															by_partitions=False, partition_length=14, 
															f_resample=f_resample, fmri_resolution_factor=1, 
															standardize_eeg=standardize_eeg, standardize_fmri=standardize_fmri,
															ind_volume_fit=ind_volume_fit, iqr_outlier=iqr,
															eeg_limit=eeg_limit, eeg_f_limit=eeg_f_limit,
															dataset=dataset)

	eeg_channels=eeg_train.shape[1]

	n_individuals_train = getattr(data_utils, "n_individuals_train_"+dataset)
	n_individuals_test = getattr(data_utils, "n_individuals_test_"+dataset)
	n_volumes = getattr(fmri_utils, "n_volumes_"+dataset)

	if(raw_eeg):
		eeg_test = eeg_train[n_individuals_train*(n_volumes)*int(f_resample*getattr(eeg_utils, "fs_"+dataset)):(n_individuals_train+n_individuals_test)*n_volumes*int(f_resample*getattr(eeg_utils, "fs_"+dataset))]
		eeg_train = eeg_train[:n_individuals_train*n_volumes*int(f_resample*getattr(eeg_utils, "fs_"+dataset))]
	else:
		eeg_test = eeg_train[n_individuals_train*n_volumes:(n_individuals_train+n_individuals_test)*n_volumes]
		eeg_train = eeg_train[:n_individuals_train*n_volumes]

	fmri_test = fmri_train[n_individuals_train*n_volumes:(n_individuals_train+n_individuals_test)*n_volumes]
	fmri_train = fmri_train[:n_individuals_train*n_volumes]

	if(verbose):
		if(file_output==None):
			print("I: Finished Loading Data")
		else:
			print("I: Finished Loading Data", file=file_output)

	eeg_train, fmri_train = data_utils.create_eeg_bold_pairs(eeg_train, fmri_train, 
															raw_eeg=raw_eeg,
															fs_sample_eeg=getattr(eeg_utils, "fs_"+dataset),
															fs_sample_fmri=f_resample,
															interval_eeg=interval_eeg, 
															n_volumes=n_volumes, 
															n_individuals=n_individuals_train,
															instances_per_individual=25)
	eeg_test, fmri_test = data_utils.create_eeg_bold_pairs(eeg_test, fmri_test, 
															raw_eeg=raw_eeg,
															fs_sample_eeg=getattr(eeg_utils, "fs_"+dataset),
															fs_sample_fmri=f_resample,
															interval_eeg=interval_eeg, 
															n_volumes=n_volumes, 
															n_individuals=n_individuals_test,
															instances_per_individual=25)

	eeg_train = np.expand_dims(eeg_train, axis=-1)
	fmri_train = np.expand_dims(fmri_train, axis=-1)
	eeg_test = np.expand_dims(eeg_test, axis=-1)
	fmri_test = np.expand_dims(fmri_test, axis=-1)

	eeg_train = eeg_train.astype('float32')
	fmri_train = fmri_train.astype('float32')
	eeg_test = eeg_test.astype('float32')
	fmri_test = fmri_test.astype('float32')

	if(verbose):
		if(file_output == None):
			print("I: Pairs Created")
		else:
			print("I: Pairs Created", file=file_output)

	return (eeg_train, fmri_train), (eeg_test, fmri_test)


def dataset_clf(dataset, n_individuals=8, mutate_bands=False, f_resample=2, raw_eeg=False, raw_eeg_resample=False, eeg_limit=False, eeg_f_limit=134, file_output=None, standardize_eeg=False, interval_eeg=10, verbose=False):

	n_individuals_train=getattr(data_utils, "n_individuals_train_"+dataset)
	n_individuals_test=getattr(data_utils, "n_individuals_test_"+dataset)
	recording_time=getattr(eeg_utils, "recording_time_"+dataset)
	eeg_limit=getattr(eeg_utils, "fs_"+dataset)>250
	eeg_f_limit=135

	if(verbose):
		if(file_output == None):
			print("I: Loading data")
		else:
			print("I: Loading data", file=file_output)

	X, y = data_utils.load_data_clf(dataset, n_individuals=n_individuals, 
									mutate_bands=mutate_bands, f_resample=f_resample, 
									raw_eeg=raw_eeg, raw_eeg_resample=raw_eeg_resample, 
									eeg_limit=eeg_limit, eeg_f_limit=eeg_f_limit, 
									recording_time=recording_time,
									standardize_eeg=standardize_eeg)

	if(verbose):
		if(file_output == None):
			print("I: Creating pairs")
		else:
			print("I: Creating pairs", file=file_output)

	X_train, y_train = data_utils.create_clf_pairs(n_individuals_train, X[:n_individuals_train*recording_time], 
											y[:n_individuals_train], 
											raw_eeg=raw_eeg,
											recording_time=recording_time, 
											interval_eeg=interval_eeg)
	X_test, y_test = data_utils.create_clf_pairs(n_individuals_test, X[:n_individuals_test*recording_time], 
												y[:n_individuals_test], 
												raw_eeg=raw_eeg,
												recording_time=recording_time,
												interval_eeg=interval_eeg)

	if(verbose):
		if(file_output == None):
			print("I: Finished loading data")
		else:
			print("I: Finished loading data", file=file_output)
	
	X_train = np.expand_dims(X_train, axis=-1)
	X_test = np.expand_dims(X_test, axis=-1)

	X_train = X_train.astype('float32')
	y_train = y_train.astype('float32')
	X_test = X_test.astype('float32')
	y_test = y_test.astype('float32')

	return (X_train, y_train), (X_test, y_test)




class Dataset_CLF_CV:
	"""
	This class is a wrapper for a dataset that returns the folds in a cross validation setting
	"""

	
	def __init__(self, dataset, mutate_bands=False, f_resample=2, raw_eeg=False, raw_eeg_resample=False, eeg_limit=False, eeg_f_limit=134, standardize_eeg=False, load=True, load_path=None, seed=None, verbose=False):
		"""
		Inputs:
			* str - dataset identifier
			* bool - mutate_bands mutate frequency domain to correspond to bands (DEPRECATED)
			* bool - f_resample resample EEG signal, used for STFT window size
			* bool - raw_eeg whether to return EEG as channel time or channel frequency time
			* bool - raw_eeg_resample if (raw EEG) should be resampled to f_resample
			* bool - eeg_limit whether to give a frequency high pass filter
			* int - eeg_f_limit frequency high pass filter value
			* bool - standardize_eeg whether or not to scale the dataset to be a Normal(0,1)
			* bool - verbose whether to print state of execution
		"""
		assert dataset in ["10", "11"], "Dataset not recognized, may not yet be implemented"

		self.n_individuals=getattr(data_utils, "n_individuals_"+dataset)
		self.recording_time=getattr(eeg_utils, "recording_time_"+dataset)
			
		self.eeg_limit=eeg_limit
		self.eeg_f_limit=eeg_f_limit
		self.interval_eeg=10
		self.seed=seed

		if(load):
			X, y = data_utils.load_data_clf(dataset, n_individuals=self.n_individuals, 
											mutate_bands=mutate_bands, f_resample=f_resample, 
											raw_eeg=raw_eeg, raw_eeg_resample=raw_eeg_resample, 
											eeg_limit=eeg_limit, eeg_f_limit=eeg_f_limit, 
											recording_time=self.recording_time,
											standardize_eeg=standardize_eeg)
			
			self.X, self.y = data_utils.create_clf_pairs(self.n_individuals, 
														X, y, raw_eeg=raw_eeg,
														recording_time=self.recording_time, 
														interval_eeg=self.interval_eeg)
			
			self.X = np.expand_dims(self.X, axis=-1)

			del X,y
			gc.collect()
		else:
			self.load(load_path)

		self.folds=self.n_individuals
		self.ind_time=(self.X.shape[0])//(self.n_individuals)
	
	
	def set_folds(self, n):
		"""
		Inputs:
			* int - number of folds, default is leave one out
		"""
		
		self.folds=n
		
	
	def split(self, fold):
		"""
		Inputs:
			* int - fold number
		Outputs:
			* tuple(tuple, tuple) - train test split
		"""
		
		assert fold < self.n_individuals, "A fold is within {0, 1, ..., N-1} for a total of N folds"
		
		step=(self.n_individuals//self.folds)
		return (np.append(self.X[:fold*self.ind_time*step], self.X[fold*self.ind_time*step+self.ind_time*step:], axis=0), 
						np.append(self.y[:fold*self.ind_time*step], self.y[fold*self.ind_time*step+self.ind_time*step:], axis=0)), \
				(self.X[fold*self.ind_time*step:fold*self.ind_time*step+self.ind_time*step], 
						 self.y[fold*self.ind_time*step:fold*self.ind_time*step+self.ind_time*step])

	
	def shuffle(self):
		"""
		Shuffle dataset

		This should only be done in a CV setting with fold < individuals, i.e. please do not do this in a LOOCV setting
		"""
		data = shuffle(self.X, self.y, random_state=self.seed)
		self.X=data[0]
		self.y=data[1]


	
	def save(self, path):
		"""
		Save data to numpy array to ease loading bottleneck
		"""
		np.save(path+"X.npy", self.X, allow_pickle=True)
		np.save(path+"y.npy", self.y, allow_pickle=True)

	
	def load(self, path):
		"""
		Load data from .npy file
		"""
		self.X = np.load(path+"X.npy", allow_pickle=True)
		self.y = np.load(path+"y.npy", allow_pickle=True)


class DatasetContrastive:

	def __init__(self, X, labels, batch=4, pairs=4, repeat_pairing=False, clf=False, seed=None):
		"""

			arguemnt clf specifies if one also gives the labels of the data in the tensorflow.data.Dataset

		"""
		self.seed=seed

		self.X = X
		self.labels = labels

		#number of pairs has to be lower than the size of the dataset
		self.pairs=X.shape[0]//2

		self.data=None
		self.y=None
		self.y1=None
		self.y2=None

		self.batch=batch
		self.clf=clf
		self.repeat_pairing=repeat_pairing
		self.tf_dataset=None

	
	def shuffle(self):
		"""
		Shuffle dataset

		This should only be done in a CV setting with fold < individuals, i.e. please do not do this in a LOOCV setting
		"""
		data = shuffle(self.X, self.labels, random_state=self.seed)
		self.X=data[0]
		self.labels=data[1]

	@property
	def pairwise(self):
		del self.data, self.y, self.y1, self.y2
		gc.collect()
		
		self.data=np.empty((self.pairs*2,2,)+self.X.shape[1:], dtype=np.float32)
		
		self.y=np.empty((self.pairs*2,2), dtype=np.float32)

		if(self.clf):
			self.y1=np.empty((self.pairs*2,2), dtype=np.float32)
			self.y2=np.empty((self.pairs*2,2), dtype=np.float32)

		instance=0

		positive=self.pairs
		negative=self.pairs
		
		while(positive>0 or negative>0):
			
			#choice is done randomly throughout the dataset
			indices = np.random.choice(np.arange(0,self.X.shape[0]), size=2, replace=False)
			i1, i2 = (indices[0], indices[1])

			if(np.all(self.labels[i1]==self.labels[i2])):
				if(positive==0):
					continue
				self.y[instance]=np.array([0.0, 1.0], dtype=np.float32)
				positive-=1
			else:
				if(negative==0):
					continue
				self.y[instance]=np.array([1.0, 0.0], dtype=np.float32)
				negative-=1

			if(self.clf):
				self.y1[instance]=self.labels[i1].astype(np.float32)
				self.y2[instance]=self.labels[i2].astype(np.float32)

			self.data[instance]=np.concatenate((self.X[i1:i1+1], self.X[i2:i2+1]), axis=0)
			instance+=1

		if(self.clf):
			return self.data, np.concatenate((np.expand_dims(self.y, axis=1), np.concatenate((np.expand_dims(self.y1,axis=1), np.expand_dims(self.y2,axis=1)), axis=1)), axis=1)

		return self.data, self.y

	def repeat(self, n, shuffle=True):
		"""
		repeat n times the dataset, this is a wrapper for the train session
		"""

		#if(not self.repeat_pairing and self.tf_dataset is not None):
		#	return self.tf_dataset

		del self.tf_dataset
		gc.collect()

		if(shuffle):
			self.tf_dataset=tf.data.Dataset.from_tensor_slices(self.pairwise).shuffle(self.X.shape[0]*self.pairs*2, seed=self.seed).batch(self.batch).repeat(n)
		else:
			self.tf_dataset=tf.data.Dataset.from_tensor_slices(self.pairwise).batch(self.batch).repeat(n)

		return self.tf_dataset