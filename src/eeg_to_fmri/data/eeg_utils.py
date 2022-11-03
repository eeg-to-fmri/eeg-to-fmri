import mne

import matplotlib.pyplot as plt

from scipy.fft import fft
from scipy.io import loadmat
from scipy.signal import cwt, ricker
from scipy import signal as scipy_signal

import numpy as np

import os
from os import listdir
from os.path import isfile, join, isdir
from pathlib import Path

import csv

home = str(Path.home())

channels_01=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'POz', 'ECG', 'AF3', 'AF4', 'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'FT9', 'FT10', 'P9', 'P10', 'PO9', 'PO10', 'O9', 'O10', 'Fpz', 'CPz']
channels_02=["C3-T7","T7-LM","LM-CP5","CP5-P7","P7-PO7","PO7-PO3","PO3-O1","O1-Oz","PO3-P3","P3-CP1","Pz-CP1","CP1-C3","Cz-C3","Fp2-Fp1","Fp1-AF3","AF4-Fp2","AF3-F3","F4-AF4","F3-F7","F8-F4","FC1-F3","F4-FC2","F7-FC5","FC6-F8","FC5-T7","T8-FC6","Cz-Fz","Fz-FC1","FC2-Fz","T8-C4","RM-T8","CP6-RM","P8-CP6","PO8-P8","PO4-PO8","O2-PO4","Oz-O2","P4-PO4","CP2-P4","CP2-Pz","C4-CP2","C4-Cz","Pz-Oz"]
channels_03=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'POz', 'ECG', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8',  'FT9', 'FT10', 'Fpz', 'CPz']
channels_04=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'POz', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'FT9', 'FT10', 'Fpz', 'CPz', 'ECG']
channels_05=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'POz', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'FT9', 'FT10', 'Fpz', 'CPz', 'ECG']
channels_10=["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16","A17","A18","A19","A20","A21","A22","A23","A24","A25","A26","A27","A28","A29","A30","A31","A32","B1","B2","B3","B4","B5","B6","B7","B8","B9","B10","B11","B12","B13","B14","B15","B16","B17","B18","B19","B20","B21","B22","B23","B24","B25","B26","B27","B28","B29","B30","B31","B32","C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12","C13","C14","C15","C16","C17","C18","C19","C20","C21","C22","C23","C24","C25","C26","C27","C28","C29","C30","C31","C32","D1","D2","D3","D4","D5","D6","D7","D8","D9","D10","D11","D12","D13","D14","D15","D16","D17","D18","D19","D20","D21","D22","D23","D24","D25","D26","D27","D28","D29","D30","D31","D32","EXG1","EXG2","EXG7","EXG8"]
channels_11=['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz', 'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'Status']
channels_coords_10_20= {"Fpz": (0.5,0.9),"Fp1": (0.40,0.88),"Fp2": (0.6,0.88),"AFz": (0.5,0.8),"AF3": (0.43,0.79),"AF4": (0.57,0.79),"AF7": (0.29,0.83),"AF8": (0.71,0.83),"Fz": (0.5,0.7),"F1": (0.41,0.7),"F2": (0.59,0.7),"F3": (0.32,0.71),"F4": (0.68,0.71),"F5": (0.25,0.725),"F6": (0.75,0.725),"F7": (0.19,0.74),"F8": (0.81,0.74),"F9": (0.12,0.78),"F10": (0.88,0.78),"FCz": (0.5,0.6),"FC1": (0.4,0.6),"FC2": (0.6,0.6),"FC3": (0.3,0.605),"FC4": (0.7,0.605),"FC5": (0.22,0.615),"FC6": (0.78,0.615),"FT7": (0.13,0.63),"FT8": (0.87,0.63),"FT9": (0.05,0.655),"FT10": (0.95,0.655),"Cz": (0.5,0.5),"C1": (0.4,0.5),"C2": (0.6,0.5),"C3": (0.3,0.5),"C4": (0.7,0.5),"C5": (0.2,0.5),"C6": (0.8,0.5),"T7": (0.1,0.5),"T8": (0.9,0.5),"CPz": (0.5,0.4),"CP1": (0.4,0.4),"CP2": (0.6,0.4),"CP3": (0.3,0.395),"CP4": (0.7,0.395),"CP5": (0.22,0.385),"CP6": (0.78,0.385),"TP7": (0.13,0.37),"TP8": (0.87,0.37),"TP9": (0.05,0.345),"TP10": (0.95,0.345),"Pz": (0.5,0.3),"P1": (0.41,0.3),"P2": (0.59,0.3),"P3": (0.32,0.29),"P4": (0.68,0.29),"P5": (0.25,0.275),"P6": (0.75,0.275),"P7": (0.19,0.26),"P8": (0.81,0.26),"P9": (0.12,0.22),"P10": (0.88,0.22),"POz": (0.5,0.2),"PO3": (0.43,0.21),"PO4": (0.57,0.21),"PO7": (0.29,0.17),"PO8": (0.71,0.17),"PO9": (0.25,0.1),"PO10": (0.75,0.1),"Oz": (0.5,0.1),"O1": (0.40,0.12),"O2": (0.6,0.12),"O9": (0.37,0.05),"O10": (0.63,0.05)}

#frequency samples of each EEG dataset
fs_01=250
fs_02=1000
fs_03=5000
fs_04=200
fs_05=200
fs_NEW=None
fs_10=2048
fs_11=512
fs_12=None
fs_13=None
fs_14=None

fs_NEW=None
recording_time_10=203
recording_time_11=90
recording_time_12=None
recording_time_13=None
recording_time_14=None

media_directory=os.environ['EEG_FMRI_DATASETS']+"/"
dataset_01="ds000001"
dataset_02="ds000116"
dataset_03="ds002158"
dataset_04="ds002336"
dataset_05="ds002338"
dataset_06="ds003768"
dataset_NEW="NEW"
dataset_10="ds004000"#schizophrenia
dataset_11="ds002778"#parkinson's uc san diego ask authors for validation -- no good results were gathered
dataset_12="ds003509"#mne.io.read_raw_eeglab parkinson's
dataset_13="ds003506"#parkinson's
dataset_14="ds003944"#schizophrenia

##########################################################################################################################
#
#											READING UTILS
#			
##########################################################################################################################
def get_eeg_instance_01(individual, path_eeg=os.environ['EEG_FMRI']+'/datasets/01/EEG/', preprocessed=True):

	individuals = sorted([f for f in listdir(path_eeg) if isdir(join(path_eeg, f))])

	individual = individuals[individual]

	if(preprocessed):
		path = path_eeg + individual + '/export/'
	else:
		path = path_eeg + individual + '/raw/'


	brainvision_files = sorted([f for f in listdir(path) if isfile(join(path, f))])

	vhdr_file = brainvision_files[1]

	complete_path = path + vhdr_file

	return mne.io.read_raw_brainvision(complete_path, preload=False, verbose=0)


def get_eeg_instance_02(individual, task=0, run=0, total_runs=3, preprocessed=True, path_eeg=os.environ['EEG_FMRI']+'/datasets/02'):

	individuals = sorted([f for f in listdir(path_eeg) if isdir(join(path_eeg, f))])
	
	path_eeg = path_eeg + '/' + individuals[individual] + '/EEG'
	
	runs = sorted([f for f in listdir(path_eeg) if isdir(join(path_eeg, f))])
	
	run = runs[task*total_runs+run]

	if(preprocessed):
		path = path_eeg + '/' + run + '/EEG_noGA.mat'
	else:
		path = path_eeg + '/' + run + '/EEG_noGA.mat'

	eeg_file = loadmat(path)
	
	return eeg_file['data_noGA'][:43,:]


def get_eeg_instance_03(individual, path_eeg=media_directory+dataset_03+"/", run="main_run-001", preprocessed=False):
	
	run_types=["main_run-001", "main_run-002",
			  "main_run-003", "main_run-004",
			  "main_run-005", "main_run-006"]
	
	assert run in run_types, dataset_03+ " contains the following recording sessions: " + str(run_types) + ", please select one."
	assert not preprocessed, "Preprocessed EEG signal is not available, only EEG events"
	
	individuals = sorted([f for f in listdir(path_eeg) if isdir(join(path_eeg, f))])[2:]

	individual = individuals[individual]

	if(preprocessed):
		path = path_eeg + "derivatives/eegprep/" + individual + "/ses-001/eeg/" + individual + "_ses-001_task-main_eeg_preproc.set"
		return mne.io.read_epochs_eeglab(path)
	else:
		path = path_eeg + individual + "/ses-001/eeg/"

		brainvision_files = sorted([f for f in listdir(path) if isfile(join(path, f))])

		vhdr_file = brainvision_files[1]

		complete_path = path + vhdr_file

		return mne.io.read_raw_brainvision(complete_path, preload=False, verbose=0)


def get_eeg_instance_04(individual, path_eeg=media_directory+dataset_04+"/derivatives/", task="eegfmriNF", preprocessed=True):

	assert task in ["eegNF", "eegfmriNF", "fmriNF", "motorloc"]
	
	individuals = sorted([f for f in listdir(path_eeg) if isdir(join(path_eeg, f))])
	
	individual = individuals[individual]
	
	path = path_eeg + individual + "/eeg_pp/"
	brainvision_files = sorted([f for f in listdir(path) if isfile(join(path, f)) and task in path+f])

	vhdr_file = brainvision_files[1]

	complete_path = path + vhdr_file

	return mne.io.read_raw_brainvision(complete_path, preload=False, verbose=0)

def get_eeg_instance_05(individual, path_eeg=media_directory+dataset_05+"/derivatives/", task="MIpost", preprocessed=True):

	assert task in ["1dNF_run-01", "1dNF_run-02", "1dNF_run-03", "MIpost", "MIpre"]
	
	individuals = sorted([f for f in listdir(path_eeg) if isdir(join(path_eeg, f))])
	individual = individuals[individual]
	
	path = path_eeg + individual + "/eeg_pp/"
	brainvision_files = sorted([f for f in listdir(path) if isfile(join(path, f)) and task in path+f])
	
	vhdr_file = brainvision_files[1]

	complete_path = path + vhdr_file

	return mne.io.read_raw_brainvision(complete_path, preload=False, verbose=0)

def get_eeg_instance_06(individual, path_eeg=media_directory+dataset_06+"/derivatives/", task="MIpost", preprocessed=True):
	raise NotImplementedError


def get_eeg_instance_NEW(individual, path_eeg=None, task=None, preprocessed=None):
	"""
	Function that reads EEG data from a <NEW> dataset

	This function should return an EEG raw brainvision object, please refer to: https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw

	Inputs:
		* int - individual, should be an integer \in [0,NUMBER_INDIVIDUALS_NEW]
		* str - path_eeg, the path specification where the individuals are listed
		* str - task, optional, depends on the dataset you are operating with
		* bool - preprocessed, specifies if one is reading preprocessed data or not, feel free to implement it as you wish
	Outputs:
		* mne.io.Raw - the EEG object
	"""
	raise NotImplementedError

def get_eeg_instance_10(individual, path_eeg=media_directory+dataset_10+"/", proposer=False, preprocessed=False):

	assert not preprocessed, "Preprocessed EEG signal is not available, only EEG events"

	individuals = sorted([f for f in listdir(path_eeg) if isdir(join(path_eeg, f))])[:]
	individual = individuals[individual]
	
	
	if(preprocessed):
		path = path_eeg + "derivatives/eegprep/" + individual + "/eeg/" + individual + "_task-main_eeg_preproc.set"
		return mne.io.read_epochs_eeglab(path)
	else:
		path = path_eeg + individual + "/eeg/"

		brainvision_files = sorted([f for f in listdir(path) if "vhdr" in f])
		
		if(proposer):
			vhdr_file = brainvision_files[0]
		else:
			vhdr_file = brainvision_files[1]
		complete_path = path + vhdr_file
		
		return mne.io.read_raw_brainvision(complete_path, preload=False, verbose=0)

def get_eeg_instance_11(individual, path_eeg=media_directory+dataset_11+"/", sess_on=True, preprocessed=False):
	
	assert not preprocessed, "Preprocessed EEG signal is not available, only EEG events"
	
	individuals = sorted([f for f in listdir(path_eeg) if isdir(join(path_eeg, f))])[1:]
	individual = individuals[individual]

	if(preprocessed):
		path = path_eeg + "derivatives/eegprep/" + individual + "/eeg/" + individual + "_task-main_eeg_preproc.set"
		return mne.io.read_epochs_eeglab(path)
	else:
		if("hc" in individual):
			session="ses-hc"
		elif("pd" in individual and sess_on):
			session="ses-on"
		else:
			session="ses-off"
		
		path = path_eeg + individual + "/" + session + "/eeg/"

		complete_path = path + individual + "_" + session + "_task-rest_eeg.bdf" 
		
		return mne.io.read_raw_bdf(complete_path, preload=False, verbose=0)

def get_eeg_instance_12(individual, path_eeg=media_directory+dataset_12+"/", sess_on=True, preprocessed=False):
	raise NotImplementedError

def get_eeg_instance_13(individual, path_eeg=media_directory+dataset_13+"/", sess_on=True, preprocessed=False):
	raise NotImplementedError
	
def get_eeg_instance_14(individual, path_eeg=media_directory+dataset_14+"/", sess_on=True, preprocessed=False):
	raise NotImplementedError
	
def get_labels_10(individuals, path_eeg=media_directory+dataset_10+"/"):
	labels=np.zeros((len(individuals), 2))#0 - hc, 1 - p
	
	participants_info_file = open(path_eeg+"participants.tsv", "r")
	
	participants_info = participants_info_file.readlines()[1:]

	for ind in range(len(participants_info)):		
		if("HC" in participants_info[ind].split("\t")[1]):
			labels[ind][0] = 1.0
		elif("P" in participants_info[ind].split("\t")[1]):
			labels[ind][1] = 1.0
			
	participants_info_file.close()

	return labels

def get_labels_11(individuals, path_eeg=media_directory+dataset_11+"/"):
	labels=np.zeros((len(individuals), 2))#0 - hc, 1 - p
	
	individuals = sorted([f for f in listdir(path_eeg) if isdir(join(path_eeg, f))])[1:len(individuals)+1]
	
	for ind in range(len(individuals)):
		if("hc" in individuals[ind]):
			labels[ind][0] = 1.0
		elif("pd" in individuals[ind]):
			labels[ind][1] = 1.0
			
	return labels


def get_labels_12(individuals, path_eeg=media_directory+dataset_12+"/"):
	raise NotImplementedError

def get_labels_13(individuals, path_eeg=media_directory+dataset_13+"/"):
	raise NotImplementedError

def get_labels_14(individuals, path_eeg=media_directory+dataset_14+"/"):
	raise NotImplementedError

def get_eeg_dataset(number_individuals=16, path_eeg=os.environ['EEG_FMRI']+'/datasets/01/EEG/', preprocessed=True):
	individuals = []

	for i in range(number_individuals):
		individuals += [get_individual(i, path_eeg=path_eeg, preprocessed=True)]

	return individuals


##########################################################################################################################
#
#											FREQUENCY UTILS
#			
##########################################################################################################################
frequency_bands = {'delta': [0.5,4], 'theta': [4,8], 'alpha': [8,13], 'beta': [13,30], 'gamma': [30, 100]}


def compute_fft(channel, fs=128, limit=False, f_limit=134):
	N = int(len(channel)/2)

	fft1 = fft(channel)

	if(limit):
		if(fft1.shape[0]>f_limit):
			return fft1[range(f_limit)]
		return np.append(fft1, np.zeros((f_limit-fft1.shape[0],), dtype=np.complex))
	return fft1[range(int(N/2))]

def raw_eeg(eeg, channel=0, fs=250):
	signal = eeg[channel][:]
	if(type(signal) is tuple):
		signal, _ = signal
		signal = signal.reshape((signal.shape[1]))
	else:
		signal = signal.reshape((signal.shape[0]))

	return signal

def stft(eeg, channel=0, window_size=2, fs=250, limit=False, f_limit=134, start_time=None, stop_time=None):
	signal = eeg[channel][:]
	if(type(signal) is tuple):
		signal, _ = signal
		signal = signal.reshape((signal.shape[1]))
	else:
		signal = signal.reshape((signal.shape[0]))


	if(start_time == None):
		start_time = 0
	if(stop_time == None):
		stop_time = len(signal)
	signal = signal[start_time:stop_time]

	t = []



	fs_window_size = int(window_size*fs)


	Z = []
	seconds = 0
	for time in range(start_time, stop_time, fs_window_size)[:-1]:
		fft1 = compute_fft(signal[time:time+fs_window_size], fs=fs, limit=limit, f_limit=f_limit)

		N = len(signal[time:time+fs_window_size])/2
		f = np.linspace (0, len(fft1), int(N/2))

		#average
		Z += [list(abs(fft1[1:]))]
		t += [seconds]
		seconds += window_size

	return f[1:], np.transpose(np.array(Z)), t


def dwt(eeg, channel=0, windows=30, fs=2.0, start_time=None, stop_time=None):

	signal = eeg[channel][:]
	if(type(signal) is tuple):
		signal, _ = signal
		signal = signal.reshape((signal.shape[1]))
	else:
		signal = signal.reshape((signal.shape[0]))


	if(start_time == None):
		start_time = 0
	if(stop_time == None):
		stop_time = len(signal)
	signal = signal[start_time:stop_time]


	return cwt(signal, scipy_signal.morlet2, np.arange(1, windows))


def mutate_stft_to_bands(Zxx, frequencies, timesteps):
	#frequency first dimension, time is second dimension
	Z_band_mutated = []

	for t in range(len(timesteps)):

		intensities = []

		for i in range(len(frequency_bands.keys())):
			intensities += [0]

		bands = dict(zip(frequency_bands.keys(), intensities))

		for f in range(len(frequencies)):
			for band in bands.keys():
				if(frequencies[f] <= frequency_bands[band][1]):
					bands[band] += Zxx[f][t]
					break

		Z_band_mutated += [list(bands.values()).copy()]

	return np.transpose(np.array(Z_band_mutated))

##########################################################################################################################
#
#											VISUALIZATION UTILS
#			
##########################################################################################################################
def plot_fft(eeg, channel=0, max_freq=30000, start_time=None, stop_time=None):
	y, _ = eeg[channel][:]
	y = y[0]
	
	fs = eeg.info['sfreq']

	if(start_time == None):
		start_time = 0
	if(stop_time == None):
		stop_time = len(y)
	
	fft1 = compute_fft(y[start_time:stop_time], fs=fs)
	
	N = int(len(y[start_time:stop_time])/2)
	f = np.linspace (0, fs, N//2)
	
	plt.figure(1)
	plt.plot (f[1:max_freq], abs (fft1)[1:max_freq])
	plt.title ('Magnitude of each frequency')
	plt.xlabel ('Frequency (Hz)')
	plt.show()


def plot_stft(eeg, channel=2, window_size=2, min_freq=None, max_freq=None, colorbar=True):
	f, Zxx, t = stft(eeg, channel=channel, fs=eeg.info['sfreq'], window_size=window_size)
	
	if(min_freq == None):
		min_freq = 0
	if(max_freq == None):
		max_freq = len(Zxx)

	Zxx = Zxx[min_freq:max_freq]
	f = f[min_freq:max_freq]

	amplitude = np.max(Zxx)

	fig, axes = plt.subplots(1,1)

	im = axes.pcolormesh(t, f, abs(Zxx), vmin=0, vmax=amplitude)

	if(colorbar):
		fig.colorbar(im)

	fig.show()

	return fig, axes