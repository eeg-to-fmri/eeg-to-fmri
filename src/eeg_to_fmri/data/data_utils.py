from eeg_to_fmri.data import eeg_utils, fmri_utils, outlier_utils

import numpy as np
from numpy import correlate

import mne
from nilearn.masking import apply_mask, compute_epi_mask
from nilearn import signal, image

from sklearn.preprocessing import normalize

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from scipy.signal import resample
from scipy.stats import zscore

import sys

n_partitions = 16
number_channels = 64
number_individuals = 16

threshold_plot_01=0.37
threshold_plot_02=0.37
threshold_plot_03=0.37
threshold_plot_04=0.37
threshold_plot_05=0.37

n_individuals_01=10
n_individuals_02=10
n_individuals_03=20
n_individuals_04=10
n_individuals_05=17
n_individuals_NEW=None
n_individuals_10=43
n_individuals_11=31
n_individuals_12=None
n_individuals_13=None
n_individuals_14=None

n_individuals_train_01 = 8
n_individuals_test_01 = 2
n_individuals_train_02 = 8
n_individuals_test_02 = 2
n_individuals_train_03 = 16
n_individuals_test_03 = 4
n_individuals_train_04 = 8
n_individuals_test_04 = 2
n_individuals_train_05 = 13
n_individuals_test_05 = 4
n_individuals_train_10 = 30
n_individuals_test_10 = 13
n_individuals_train_11 = 25
n_individuals_test_11 = 6
n_individuals_train_12 = None
n_individuals_test_12 = None
n_individuals_train_13 = None
n_individuals_test_13 = None
n_individuals_train_14 = None
n_individuals_test_14 = None


#############################################################################################################
#
#                                 LOAD DATA FUNCTION                     
#
#############################################################################################################

def load_data(instances, raw_eeg=False, n_voxels=None, bold_shift=3, n_partitions=16, by_partitions=True, partition_length=None, f_resample=2, mutate_bands=False, eeg_limit=False, eeg_f_limit=134, fmri_resolution_factor=4, standardize_eeg=True, standardize_fmri=True, ind_volume_fit=True, iqr_outlier=True, roi=None, roi_ica_components=None, dataset="01"):

    #Load Data
    eeg, bold, scalers = get_data(instances,
                                    raw_eeg=raw_eeg,
                                    n_voxels=n_voxels, bold_shift=bold_shift, n_partitions=n_partitions, 
                                    by_partitions=by_partitions, partition_length=partition_length,
                                    f_resample=f_resample, mutate_bands=mutate_bands,
                                    fmri_resolution_factor=fmri_resolution_factor,
                                    standardize_fmri=standardize_fmri,
                                    ind_volume_fit=ind_volume_fit,
                                    standardize_eeg=standardize_eeg,
                                    iqr_outlier=iqr_outlier,
                                    TR=getattr(fmri_utils, "TR_"+dataset),
                                    eeg_limit=eeg_limit, eeg_f_limit=eeg_f_limit,
                                    dataset=dataset)

    return eeg, bold, scalers


def load_data_clf(dataset, n_individuals=8, mutate_bands=False, f_resample=2, raw_eeg=False, raw_eeg_resample=False, eeg_limit=False, eeg_f_limit=134, recording_time=90, standardize_eeg=False):

    return get_data_classification(list(range(n_individuals)), dataset,
                                f_resample=f_resample,
                                raw_eeg=raw_eeg,
                                raw_eeg_resample=raw_eeg_resample,
                                recording_time=recording_time,
                                mutate_bands=mutate_bands,
                                eeg_limit=eeg_limit,
                                eeg_f_limit=eeg_f_limit,
                                standardize_eeg=standardize_eeg)

"""
"""

def get_data(individuals, raw_eeg=False, raw_eeg_resample=False, eeg_resample=2.160, bold_shift=3, n_partitions=16, by_partitions=True, partition_length=None, n_voxels=None, TR=2.160, f_resample=2, mutate_bands=False, eeg_limit=False, eeg_f_limit=134, fmri_resolution_factor=5, standardize_eeg=True, standardize_fmri=True, ind_volume_fit=True, iqr_outlier=True, dataset="01"):
    TR = 1/TR

    X = []
    y = []
    fmri_scalers = []

    #setting mask and fMRI signals
    individuals_imgs = getattr(fmri_utils, "get_individuals_paths_"+dataset)(resolution_factor=fmri_resolution_factor, number_individuals=len(individuals))
    recording_time = np.amin([getattr(fmri_utils, "n_volumes_"+dataset), individuals_imgs[0].shape[-1]])
    fmri_volumes = np.empty((len(individuals)*(recording_time),) + individuals_imgs[0].get_fdata()[:,:,:,0].shape)
    j = 0
    #clean fMRI signal
    for i in range(len(individuals_imgs)):
        individuals_imgs[i] = individuals_imgs[i].get_fdata()
        
        if(iqr_outlier):
            initial_j=j
            iqr = outlier_utils.IQR()
            iqr.fit(individuals_imgs[i][:,:,:,bold_shift:])
            individuals_imgs[i] = iqr.transform(individuals_imgs[i][:,:,:,bold_shift:recording_time+bold_shift], channels_last=True)
        else:
            individuals_imgs[i] = individuals_imgs[i][:,:,:,bold_shift:recording_time+bold_shift]
        scaler = StandardScaler(copy=True)
        if(not ind_volume_fit):
            reshaped_individual = individuals_imgs[i].flatten().reshape(-1,1)
            scaler.fit(reshaped_individual)

        for volume in range(individuals_imgs[i].shape[-1]):

            volume_shape = individuals_imgs[i][:,:,:,volume].shape

            reshaped_volume = individuals_imgs[i][:,:,:,volume].flatten().reshape(-1, 1)
            
            if(ind_volume_fit):
                scaled_volume = scaler.fit_transform(reshaped_volume).reshape((1,) + volume_shape)
            elif(standardize_fmri):
                scaled_volume = scaler.transform(reshaped_volume).reshape((1,) + volume_shape)
            else:
                scaled_volume = reshaped_volume.reshape((1,) + volume_shape)
            
            fmri_volumes[j] = scaled_volume
            j += 1

        if(iqr_outlier):
            fmri_volumes[initial_j:j] = iqr.inverse_transform(fmri_volumes[initial_j:j], channels_last=False)

    individuals_imgs = fmri_volumes
    individuals_eegs = None
       
    for individual in individuals:
        eeg = getattr(eeg_utils, "get_eeg_instance_"+dataset)(individual)
        
        if(dataset=="02"):
            len_channels=len(eeg)
            fs_sample = getattr(eeg_utils, "fs_"+dataset)
        else:
            fs_sample = eeg.info['sfreq']
            len_channels = len(eeg.ch_names)
        
        x_instance = []
        #eeg
        for channel in range(len_channels):
            if(raw_eeg):
                x = eeg_utils.raw_eeg(eeg, channel=channel)
                if(raw_eeg_resample):
                    x = resample(x, int((len(x)*(1/eeg_resample))/fs_sample))
                x_instance += [x]
            else:
                f, Zxx, t = eeg_utils.stft(eeg, channel=channel, window_size=f_resample, fs=getattr(eeg_utils, "fs_"+dataset), limit=eeg_limit, f_limit=eeg_f_limit)
                if(mutate_bands):
                    Zxx = eeg_utils.mutate_stft_to_bands(Zxx, f, t)
                x_instance += [Zxx]
            
        if(standardize_eeg):
            x_instance = zscore(np.array(x_instance))
        else:
            x_instance = np.array(x_instance)

        if(not type(individuals_eegs) is np.ndarray):
            if(raw_eeg):#x_instance.shape[0]==len(eeg_utils.channels)
                individuals_eegs = np.empty((0,) +(x_instance.shape[0],))
            else:#x_instance.shape[0]==len(eeg_utils.channels) and x_instance.shape[1]==freq_limit
                individuals_eegs = np.empty((0,) + (x_instance.shape[0], x_instance.shape[1]))

        if(raw_eeg_resample):#placeholder because eeg was already resampled
            fs_sample=1
            f_resample=1

        if(raw_eeg):
            individuals_eegs = np.vstack((individuals_eegs, np.transpose(x_instance[:,int(((bold_shift))*fs_sample*f_resample):int(((recording_time+bold_shift))*fs_sample*f_resample)], (1,0))))
        else:
            individuals_eegs = np.vstack((individuals_eegs, np.transpose(x_instance, (2,0,1))[bold_shift:recording_time+bold_shift]))

    #return individuals_eegs, individuals_imgs, mask, fmri_scalers
    return individuals_eegs, individuals_imgs, fmri_scalers


def get_data_classification(individuals, dataset, raw_eeg=False, raw_eeg_resample=False, eeg_resample=2, f_resample=2, mutate_bands=False, eeg_limit=False, eeg_f_limit=134, recording_time=90, standardize_eeg=True):
    individuals_eegs = None
    
    for individual in individuals:
        #eeg = getattr(eeg_utils, "get_eeg_instance_"+dataset)(individual)
        eeg = getattr(eeg_utils, "get_eeg_instance_"+dataset)(individual)

        if(dataset=="02"):
            len_channels=len(eeg)
            fs_sample = getattr(eeg_utils, "fs_"+dataset)
        else:
            fs_sample = eeg.info['sfreq']
            len_channels = len(eeg.ch_names)


        x_instance = []
        #eeg
        for channel in range(len_channels):
            if(raw_eeg):
                x = eeg_utils.raw_eeg(eeg, channel=channel)
                if(raw_eeg_resample):
                    x = resample(x, int((len(x)*(1/eeg_resample))/fs_sample))
                x_instance += [x]
            else:
                f, Zxx, t = eeg_utils.stft(eeg, channel=channel, window_size=f_resample, fs=getattr(eeg_utils, "fs_"+dataset), limit=eeg_limit, f_limit=eeg_f_limit)
                if(mutate_bands):
                    Zxx = eeg_utils.mutate_stft_to_bands(Zxx, f, t)
                x_instance += [Zxx]

        if(standardize_eeg):
            x_instance = zscore(np.array(x_instance))
        else:
            x_instance = np.array(x_instance)

        if(not type(individuals_eegs) is np.ndarray):
            if(raw_eeg):
                individuals_eegs = np.empty((globals()["n_individuals_"+dataset]*recording_time,) +(x_instance.shape[0],))
            else:
                individuals_eegs = np.empty((globals()["n_individuals_"+dataset]*recording_time,) + (x_instance.shape[0], x_instance.shape[1]))

        if(raw_eeg_resample):#placeholder because eeg was already resampled
            fs_sample=1
            f_resample=1

        x_instance=x_instance[:132]#number of ECG channels differ for some individuals
        if(raw_eeg):
            #individuals_eegs = np.vstack((individuals_eegs, np.transpose(x_instance[:,:int(((recording_time))*fs_sample*f_resample)], (1,0))))
            individuals_eegs[individual*recording_time:individual*recording_time+recording_time] = np.transpose(x_instance[:,:int(((recording_time))*fs_sample*f_resample)], (1,0))
        else:
            #individuals_eegs = np.vstack((individuals_eegs, np.transpose(x_instance, (2,0,1))[:recording_time]))
            individuals_eegs[individual*recording_time:individual*recording_time+recording_time] = np.transpose(x_instance, (2,0,1))[:recording_time]

    return individuals_eegs, getattr(eeg_utils, "get_labels_"+dataset)(individuals)



#16 - corresponds to a 20 second length signal with 10 time points
#32 - corresponds to a 10 second length signal with 5 time points
#individuals is a list of indexes until the maximum number of individuals
def get_data_roi(individuals, raw_eeg=False, masker=None, start_cutoff=3, bold_shift=3, n_partitions=16, by_partitions=True, partition_length=None, n_voxels=None, f_resample=2, roi=None, roi_ica_components=None):
    TR = 1/2.160

    X = []
    y = []


    #setting ICA
    if(roi != None and roi_ica_components != None):
        individuals_imgs = fmri_utils.get_individuals_paths()
        roi_extraction = fmri_utils.roi_time_series()
        roi_extraction._set_ICA(individuals_imgs, n_components=roi_ica_components)

    for individual in individuals:
        eeg = eeg_utils.get_eeg_instance(individual)
        x_instance = []

        #eeg
        for channel in range(len(eeg.ch_names)):
            f, Zxx, t = eeg_utils.stft(eeg, channel=channel, window_size=f_resample) 
            Zxx_mutated = eeg_utils.mutate_stft_to_bands(Zxx, f, t)

            x_instance += [Zxx_mutated]

        x_instance = np.array(x_instance)

        #fmri
        if(roi != None and roi_ica_components != None):
            fmri_masked_instance = roi_extraction.get_ROI_time_series(individuals_imgs[individual], component=roi)
        else:
            fmri_instance = fmri_utils.get_fmri_instance_img(individual)
            fmri_masked_instance, _ = fmri_utils.get_masked_epi(fmri_instance, masker)

        fmri_resampled = []
        #build resampled BOLD signal
        if(n_voxels == None):
            n_voxels = fmri_masked_instance.shape[1]

        for voxel in range(n_voxels):
            voxel = fmri_utils.get_voxel(fmri_masked_instance, voxel=voxel)
            voxel_resampled = resample(voxel, int((len(voxel)*(1/f_resample))/TR))
            fmri_resampled += [voxel_resampled]

        fmri_resampled = np.array(fmri_resampled)

        if(by_partitions):

            for partition in range(n_partitions):
                start_eeg = start_cutoff + int((fmri_resampled.shape[1]-start_cutoff-bold_shift)/n_partitions)*partition
                end_eeg = start_cutoff + int((fmri_resampled.shape[1]-start_cutoff-bold_shift)/n_partitions)*partition + int((fmri_resampled.shape[1]-start_cutoff-bold_shift)/n_partitions)

                start_bold = start_eeg+bold_shift
                end_bold = end_eeg+bold_shift

                X += [x_instance[:,:,start_eeg:end_eeg]]

                y += list(fmri_resampled[:,start_bold:end_bold].reshape(1, fmri_resampled[:,start_bold:end_bold].shape[0], fmri_resampled[:,start_bold:end_bold].shape[1]))
        else:
            total_partitions = fmri_resampled.shape[1]//partition_length
            for partition in range(total_partitions):

                start_eeg = start_cutoff + int((fmri_resampled.shape[1]-start_cutoff-bold_shift)/(total_partitions))*partition
                end_eeg = start_cutoff + int((fmri_resampled.shape[1]-start_cutoff-bold_shift)/(total_partitions))*partition + int((fmri_resampled.shape[1]-start_cutoff-bold_shift)/(total_partitions))

                start_bold = start_eeg+bold_shift
                end_bold = end_eeg+bold_shift

                X += [x_instance[:,:,start_eeg:end_eeg]]

                y += list(fmri_resampled[:,start_bold:end_bold].reshape(1, fmri_resampled[:,start_bold:end_bold].shape[0], fmri_resampled[:,start_bold:end_bold].shape[1]))

    X = np.array(X)
    y = np.array(y)

    return X, y

def create_eeg_bold_pairs(eeg, bold, raw_eeg=False, fs_sample_eeg=250, fs_sample_fmri=2, interval_eeg=2, n_volumes=300, n_individuals=10, instances_per_individual=16):
    if(raw_eeg):
        x_eeg = np.empty((n_individuals*(n_volumes-interval_eeg),)+(eeg.shape[1],int(interval_eeg*fs_sample_eeg*fs_sample_fmri)))
    else:
        x_eeg = np.empty((n_individuals*(n_volumes-interval_eeg),)+eeg.shape[1:]+(interval_eeg,))
    x_bold = np.empty((n_individuals*(n_volumes-interval_eeg),)+bold.shape[1:])

    for individual in range(n_individuals):
        for index_volume in range(individual*(n_volumes), individual*(n_volumes)+n_volumes-interval_eeg):#the last observation is missing?
            if(raw_eeg):
                x_eeg[index_volume-individual*interval_eeg] = np.transpose(eeg[int((index_volume)*fs_sample_eeg*fs_sample_fmri):int((index_volume+interval_eeg)*fs_sample_eeg*fs_sample_fmri)], (1,0))
            else:
                if(np.transpose(eeg[index_volume:index_volume+interval_eeg], (1,2,0)).shape[-1]!=interval_eeg):
                    continue
                x_eeg[index_volume-individual*interval_eeg] = np.transpose(eeg[index_volume:index_volume+interval_eeg], (1,2,0))
            x_bold[index_volume-individual*interval_eeg] = bold[index_volume+interval_eeg]
        
    return x_eeg, x_bold





def create_clf_pairs(n_individuals, data, labels, raw_eeg=False, recording_time=90, interval_eeg=10):
    """
    Inputs:
        * n_individuals - int
        * data - np.ndarray(T, channels, freqs)
        * labels - np.ndarray(individuals, 2)
        * recording_time - int
        * interval_eeg - int
    """
    if(raw_eeg):
        X = np.empty(((n_individuals*recording_time)//interval_eeg, data.shape[1], interval_eeg))
    else:
        X = np.empty(((n_individuals*recording_time)//interval_eeg, data.shape[1], data.shape[2], interval_eeg))
    y = np.empty(((n_individuals*recording_time)//interval_eeg, 2))
    
    i = 0
    for ind in range(n_individuals):
        for time in range(0, recording_time, interval_eeg):
            if((ind*recording_time)+(time+interval_eeg) < (ind+1)*recording_time):
                if(raw_eeg):
                    X[i] = np.transpose(data[(ind*recording_time)+time:(ind*recording_time)+(time+interval_eeg)], (1,0))
                else:
                    X[i] = np.transpose(data[(ind*recording_time)+time:(ind*recording_time)+(time+interval_eeg)], (1,2,0))
                y[i] = labels[ind]
                i+=1
    
    return X[:i], y[:i]
#############################################################################################################
#
#                                 STANDARDIZE DATA FUNCTION                              
#
#############################################################################################################

def standardize(eeg, bold, eeg_scaler=None, bold_scaler=None):
    #shape = (n_samples, n_features)
    eeg_reshaped = eeg.reshape((eeg.shape[0], eeg.shape[1]*eeg.shape[2]*eeg.shape[3]*eeg.shape[4]))
    bold_reshaped = bold.reshape((bold.shape[0], bold.shape[1]*bold.shape[2]*bold.shape[3]))
    
    if(eeg_scaler == None):
        eeg_scaler = StandardScaler()
        eeg_scaler.fit(eeg_reshaped)
        
    if(bold_scaler == None):
        bold_scaler = StandardScaler()
        bold_scaler.fit(bold_reshaped)

    eeg_reshaped = eeg_scaler.transform(eeg_reshaped)
    bold_reshaped = bold_scaler.transform(bold_reshaped)

    eeg_reshaped = eeg_reshaped.reshape((eeg.shape))
    bold_reshaped = bold_reshaped.reshape((bold.shape))
    
    return eeg_reshaped, bold_reshaped, eeg_scaler, bold_scaler


"""
inverse_instance_scaler - perform inverse operation to get original fMRI signal of an instance
"""
def inverse_instance_scaler(instance, data_scaler):
    
    instance = np.swapaxes(instance, 0, 1)
    
    instance = data_scaler.inverse_transform(instance)
    
    return np.swapaxes(instance, 0, 1)

"""
inverse_set_scaler - perform inverse operation to get original fMRI signals of a dataset
"""
def inverse_set_scaler(data, data_scalers, n_partitions=25):
    unscaled_data = []
    
    for i in range(len(data)):
        
        scaler_index = i//n_partitions
        
        unscaled_data += [inverse_instance_scaler(data[i], data_scalers[scaler_index])]
        
    return np.array(unscaled_data)