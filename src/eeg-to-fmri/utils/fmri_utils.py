import matplotlib.pyplot as plt

from nilearn import plotting
from nilearn import image
from nilearn import _utils
from nilearn.input_data import NiftiMasker
from nilearn.decomposition import CanICA
from nilearn.masking import apply_mask, compute_epi_mask, compute_multi_epi_mask, _apply_mask_fmri, unmask
from nilearn.image import smooth_img, index_img, iter_img, clean_img, math_img, mean_img, new_img_like

import numpy as np

from scipy.signal import resample

import os
from os import listdir
from os.path import isfile, join, isdir
from pathlib import Path

home = str(Path.home())

TR_01=2.160
TR_02=2.000
TR_03=1.280
TR_04=2.000
TR_05=1.000
TR_NEW=None

fmri_shape_01=(64,64,30)
fmri_shape_02=(64,64,32)
fmri_shape_03=(64,64,30)
fmri_shape_04=(64,64,30)
fmri_shape_05=(64,64,30)

bold_shift_01=3
bold_shift_02=3
bold_shift_03=6
bold_shift_04=3
bold_shift_05=6

n_volumes_01=300-bold_shift_01
n_volumes_02=170-bold_shift_02
n_volumes_03=370-bold_shift_03
n_volumes_04=180-bold_shift_04
n_volumes_05=300-bold_shift_05

media_directory=os.environ['EEG_FMRI_DATASETS']+"/"
dataset_01="ds000001"
dataset_02="ds000116"
dataset_03="ds002158"
dataset_04="ds002336"
dataset_05="ds002338"
dataset_NEW="NEW"

##########################################################################################################################
#
#                         READING UTILS
#       
##########################################################################################################################
def get_fmri_instance(individual=0, path_fmri=os.environ['EEG_FMRI']+'/datasets/01/fMRI/'):

    individuals = sorted([f for f in listdir(path_fmri) if isdir(join(path_fmri, f))])

    individual = individuals[individual]

    fmri_file = '/3_nw_mepi_rest_with_cross.nii.gz'

    complete_path = path_fmri + individual + fmri_file

    mask_img = compute_epi_mask(complete_path)

    return apply_mask(complete_path, mask_img)


def get_fmri_instance_img(individual=0, path_fmri=os.environ['EEG_FMRI']+'/datasets/01/fMRI/'):

    individuals = sorted([f for f in listdir(path_fmri) if isdir(join(path_fmri, f))])

    individual = individuals[individual]

    fmri_file = '/3_nw_mepi_rest_with_cross.nii.gz'

    complete_path = path_fmri + individual + fmri_file

    return image.load_img(complete_path)

def get_population_mask(path_fmri=os.environ['EEG_FMRI']+'/datasets/01/fMRI/'):

    individuals = sorted([f for f in listdir(path_fmri) if isdir(join(path_fmri, f))])

    individuals_images = []
    
    target_affine = image.load_img(path_fmri + individuals[0] + '/3_nw_mepi_rest_with_cross.nii.gz').affine
    target_shape = image.load_img(path_fmri + individuals[0] + '/3_nw_mepi_rest_with_cross.nii.gz').shape
    target_shape = (target_shape[0], target_shape[1], target_shape[2])
    
    for individual in individuals:
        fmri_file = '/3_nw_mepi_rest_with_cross.nii.gz'
        individual_path = path_fmri + individual + fmri_file
        
        if(image.load_img(individual_path).affine[0][-1] != 0.0):
            
            fmri_image = image.resample_img(image.load_img(individual_path), target_affine=target_affine, target_shape=target_shape)
            
            individuals_images += [fmri_image]

    concatenated_imgs = image.concat_imgs(individuals_images)

    return NiftiMasker(compute_multi_epi_mask(individuals_images), standardize=True).fit(concatenated_imgs)



def get_individuals_ids(path_fmri=os.environ['EEG_FMRI']+'/datasets/01/fMRI/'):

    individuals = sorted([f for f in listdir(path_fmri) if isdir(join(path_fmri, f))])

    return individuals


def get_individuals_paths_01(path_fmri=os.environ['EEG_FMRI']+'/datasets/01/fMRI/', resolution_factor = 5, number_individuals=10):
    
    fmri_individuals = []
    file_individuals = sorted([f for f in listdir(path_fmri) if isdir(join(path_fmri, f))])

    target_shape = image.load_img(path_fmri + file_individuals[0] + '/3_nw_mepi_rest_with_cross.nii.gz').shape
    target_shape = (int(target_shape[0]/resolution_factor), 
                    int(target_shape[1]/resolution_factor), 
                    int(target_shape[2]/resolution_factor))
    
    for i in range(number_individuals):
        
        individual = file_individuals[i]

        fmri_file = '/3_nw_mepi_rest_with_cross.nii.gz'

        individual_path = path_fmri + individual + fmri_file
        
        img = image.load_img(individual_path)
        
        #scale affine accordingly
        off_set = img.affine[:,3]
        new_affine = img.affine*resolution_factor
        new_affine[:,3] = off_set
        
        fmri_image = image.resample_img(img, 
                                        target_affine=new_affine,
                                        target_shape=target_shape,
                                        interpolation='nearest')

        fmri_individuals += [fmri_image]

    return fmri_individuals


def get_individuals_paths_02(path_fmri=os.environ['EEG_FMRI']+"/datasets/02/", task=1, run=1, resolution_factor = 1, number_individuals=10):
    
    task_run = "task" + '%03d' % (task,) + "_run" + '%03d' % (run,)
    
    fmri_individuals = []
    
    dir_individuals = sorted([f for f in listdir(path_fmri) if isdir(join(path_fmri, f))])
    
    #target_shape = image.load_img(path_fmri + file_individuals[0] + '/3_nw_mepi_rest_with_cross.nii.gz').shape
    #target_shape = (int(target_shape[0]/resolution_factor), 
    #      int(target_shape[1]/resolution_factor), 
    #      int(target_shape[2]/resolution_factor))
    
    affine = np.zeros((4,4))
    
    for i in range(number_individuals):
        individual_path = path_fmri + dir_individuals[i] + "/BOLD/" + task_run + "/bold.nii.gz"
        
        img = image.load_img(individual_path)
        
        affine+=img.affine
        shape=img.shape[:-1]
        
        #fmri_image = image.resample_img(img, 
        #           target_affine=new_affine,
        #           target_shape=target_shape,
        #           interpolation='nearest')

        fmri_individuals += [img]
    
    affine/=number_individuals
    
    #scale affine accordingly
    off_set = affine[:,3]
    new_affine = affine*resolution_factor
    new_affine[:,3] = off_set
    new_shape = (int(shape[0]/resolution_factor),
              int(shape[1]/resolution_factor),
              int(shape[2]/resolution_factor))
    
    for img in range(len(fmri_individuals)):
        fmri_individuals[img] = image.resample_img(fmri_individuals[img], 
                                                    target_affine=new_affine,
                                                    target_shape=new_shape,
                                                    interpolation='nearest')
    
    return fmri_individuals


def get_individuals_paths_03(path_fmri=media_directory+dataset_03+"/", 
                            resolution_factor = 5, 
                            number_individuals=20,
                            run="main_run-001",
                            downsample=True,
                            downsample_shape=(64,64,30)):
    
    run_types=["main_run-001", "main_run-002",
              "main_run-003", "main_run-004",
              "main_run-005", "main_run-006"]
    
    dir_individuals = sorted([f for f in listdir(path_fmri) if isdir(join(path_fmri, f))])[2:]
    
    assert number_individuals <= len(dir_individuals), dataset_03+ " contains a total of 20 individuals, " + str(number_individuals) + " were requested."
    assert run in run_types, dataset_03+ " contains the following recording sessions: " + str(run_types) + ", please select one."
    
    fmri_individuals=[]

    if(downsample):
        from layers import fft
        dct=None
        idct=None
    
    for i in range(number_individuals):
        individual_path = path_fmri + dir_individuals[i] + "/ses-001/func/"
        file_path= individual_path + dir_individuals[i] + "_ses-001_task-" + run + "_bold.nii.gz"

        fmri_individuals += [image.load_img(file_path)]

        if(downsample):
            img = np.swapaxes(np.swapaxes(np.swapaxes(fmri_individuals[-1].get_fdata(), 0, 3), 1,2), 1,3)
            if(dct is None and idct is None):
                dct = fft.DCT3D(*img.shape[1:])
                idct = fft.iDCT3D(*downsample_shape)
            fmri_individuals[-1] = image.new_img_like(fmri_individuals[-1], 
                                                        np.swapaxes(np.swapaxes(np.swapaxes(idct(dct(img).numpy()[:, :downsample_shape[0], :downsample_shape[1], :downsample_shape[2]]).numpy(), 0, 3), 0,2), 0,1))

    return fmri_individuals


def get_individuals_paths_04(path_fmri=media_directory+dataset_04+"/", number_individuals=None, task="eegfmriNF", downsample=True, downsample_shape=(64,64,30), resolution_factor=None):
    
    assert task in ["eegNF", "eegfmriNF", "fmriNF", "motorloc"]
    
    fmri_individuals = []
    dir_individuals = sorted([f for f in listdir(path_fmri) if isdir(join(path_fmri, f)) and "sub" in path_fmri+f])
    
    if(downsample):
        import sys
        sys.path.append("..")
        from layers import fft
        dct=None
        idct=None
        
    for i in range(number_individuals):
        task_file=sorted([f for f in listdir(path_fmri+dir_individuals[i]+'/func/') if isfile(path_fmri+dir_individuals[i]+'/func/'+f) and task in path_fmri+dir_individuals[i]+'/func/'+f])
        if(not len(task_file)):
            continue
        file_path= path_fmri+dir_individuals[i]+'/func/' + task_file[1]
        
        fmri_individuals += [image.load_img(file_path)]

        if(downsample):
            img = np.swapaxes(np.swapaxes(np.swapaxes(fmri_individuals[-1].get_fdata(), 0, 3), 1,2), 1,3)
            if(dct is None and idct is None):
                dct = fft.DCT3D(*img.shape[1:])
                idct = fft.iDCT3D(*downsample_shape)
            fmri_individuals[-1] = image.new_img_like(fmri_individuals[-1], 
                                                        np.swapaxes(np.swapaxes(np.swapaxes(idct(dct(img).numpy()[:, :downsample_shape[0], :downsample_shape[1], :downsample_shape[2]]).numpy(), 0, 3), 0,2), 0,1))
            
    return fmri_individuals
    
def get_individuals_paths_05(path_fmri=media_directory+dataset_05+"/", number_individuals=None, task="MIpost", downsample=True, downsample_shape=(64,64,30), resolution_factor=None):

    assert task in ["1dNF_run-01", "1dNF_run-02", "1dNF_run-03", "MIpost", "MIpre"]
    
    fmri_individuals = []
    dir_individuals = sorted([f for f in listdir(path_fmri) if isdir(join(path_fmri, f)) and "sub" in path_fmri+f])
    
    if(downsample):
        import sys
        sys.path.append("..")
        from layers import fft
        dct=None
        idct=None

    for i in range(number_individuals):
        
        task_file=sorted([f for f in listdir(path_fmri+dir_individuals[i]+'/func/') if isfile(path_fmri+dir_individuals[i]+'/func/'+f) and task in path_fmri+dir_individuals[i]+'/func/'+f])
        if(not len(task_file)):
            continue
        file_path= path_fmri+dir_individuals[i]+'/func/' + task_file[1]
        
        fmri_individuals += [image.load_img(file_path)]

        if(downsample):
            img = np.swapaxes(np.swapaxes(np.swapaxes(fmri_individuals[-1].get_fdata(), 0, 3), 1,2), 1,3)
            if(dct is None and idct is None):
                dct = fft.DCT3D(*img.shape[1:])
                idct = fft.padded_iDCT3D(*(downsample_shape[:2]+(img.shape[3],)+downsample_shape))
            fmri_individuals[-1] = image.new_img_like(fmri_individuals[-1], 
                                                        np.swapaxes(np.swapaxes(np.swapaxes(idct(dct(img).numpy()[:, :downsample_shape[0], :downsample_shape[1], :]).numpy(), 0, 3), 0,2), 0,1))
            
    return fmri_individuals

def get_individuals_paths_NEW(path_fmri=None, resolution_factor=None, number_individuals=None):
    """
        Function that reads fMRI data from a <NEW> dataset

        This function should return the fMRI instances (nibabel.nifti1.Nifti1Image) of a dataset, please refer to: https://nipy.org/nibabel/reference/nibabel.nifti1.html#nibabel.nifti1.Nifti1Image

        Inputs:
            * str - path_fmri, the path specification where the individuals are listed
            * float - resolution_factor, specifies the resolution of the fMRI, optional argument to implement, but mandatory as arg
            * int - number_individuals, should be an integer \in [0,NUMBER_INDIVIDUALS_NEW]
        Optional inputs to add:
            * str - task, specifies the task of the dataset
            * bool - downsample, for some datasets one might want to mutate the resolution to a desirable one
        Outputs:
            * list(nibabel.nifti1.Nifti1Image) - the list of fMRI instances, size of list is equal to number_individuals
    """

    raise NotImplementedError

##########################################################################################################################
#
#                         FMRI UTILS
#       
##########################################################################################################################
# masked_data shape is (timepoints, voxels). We can plot the first 150
# timepoints from two voxels
def get_voxel(masked_fmri, voxel=0):
    return masked_fmri[:masked_fmri.shape[0], voxel]


def get_resampled_bold(voxel, new_TR=2, TR=2.160):
    return resample(voxel, int((len(voxel)*(1/new_TR))/TR))

def get_masked_epi(fmri_instances, masker=None, smooth_factor=10, threshold="80%"):
    if(masker == None):
        if(isinstance(fmri_instances, list)):
            img_epi = image.mean_img(fmri_instances)
            img_epi = image.smooth_img(img_epi, smooth_factor)
            img_epi = image.threshold_img(img_epi, threshold=threshold)

            masker = NiftiMasker()
            masker.fit(img_epi)

            masked_instances = []

            for instance in fmri_instances:
                print(instance.data)
                exit()
                masked_instances += [apply_mask(instance, masker.mask_img_)]

            return masked_instances, masker
        else:
            masker = compute_epi_mask(fmri_instances)

    return apply_mask(fmri_instances, masker), masker

def get_inverse_masked_epi(fmri_masked, masker):
    return masker.inverse_transform(fmri_masked)


"""
get_nifti_from_voxels - transforms voxels 2D to a nifti image
"""
def get_nifti_from_voxels(voxels, mask):
    return unmask(np.swapaxes(voxels, 0, 1), mask)

"""
get_nifti_from_set - transforms a set of voxels 2D instances to a list of nifti images
"""
def get_nifti_from_set(data, mask):
    
    nifti_intances = []
    
    for instance in data:
        nifti_intances += [get_nifti_from_voxels(instance, mask)]
        
    return nifti_intances

##########################################################################################################################
#
#                         EXTRACTION OF ROI TIME SERIES
#       
##########################################################################################################################
###### Canonical ICA

#when its a population of n individuals
#imgs=[complete_path_ind_1, complete_path_ind_2, ..., complete_path_ind_n]
def _apply_mask(imgs, mask_img):
    mask_img = _utils.check_niimg_3d(mask_img)

    mask_img = _utils.check_niimg_3d(mask_img)
    mask = mask_img.get_data()
    mask = _utils.as_ndarray(mask, dtype=bool)

    mask_img = new_img_like(mask_img, mask, mask_img.affine)

    return _apply_mask_fmri(imgs, mask_img, dtype='f', smoothing_fwhm=None, ensure_finite=True)


class roi_time_series:
    def __init__(self, canica=None):
        self.canica = None

    def _set_ICA(self, imgs, n_components=20, verbose=0):
        self.canica = CanICA(n_components=n_components, smoothing_fwhm=6.,
                            memory="nilearn_cache", memory_level=2,
                            threshold=3., verbose=verbose, random_state=0)
        self._fit_ICA(imgs)

    def _fit_ICA(self, imgs):
        self.canica.fit(imgs)

    def get_ROI_time_series(self, imgs, component=0, n_components=20, verbose=False):

        #smooth image
        fmri_original = image.load_img(imgs)
        fmri_img = image.smooth_img(fmri_original, fwhm=6)

        #perform ICA and get components
        if(self.canica == None):
            if(verbose):
                print("New ICA computation")
            self._set_ICA(fmri_img, n_components=n_components)

        components_img = self.canica.components_img_

        #build masker
        roi_masker = NiftiMasker(mask_img=image.index_img(components_img, component),
                                standardize=True,
                                memory="nilearn_cache",
                                smoothing_fwhm=8)

        return _apply_mask(imgs, roi_masker.mask_img)
