# From EEG to fMRI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


[![Code: Documentation](https://img.shields.io/badge/code-documentation-green)](https://dcalhas.github.io/eeg_to_fmri/DOCUMENTATION.html)

## Setup

Ideally, your machine has a GPU and is running Linux.

First of all, please install [anaconda](https://www.anaconda.com/) at ```$HOME/anaconda3/```. To setup the environment for this repository, please run the following commands:

```shell
git clone git@github.com:DCalhas/eeg_to_fmri.git
cd eeg_to_fmri
```

Download [cudnn](https://developer.nvidia.com/cudnn):

```shell
wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.0.1/cudnn-11.0-linux-x64-v8.0.1.13.tgz
```

Run the configuration file:

```shell
./config.sh
```

Please make sure to set the path to the datasets directory correclty. This path is stored in an environment variable, so every time you activate the environment, the variable is set and used in the code as os.environ['EEG_FMRI_DATASETS'].

## How do I test this research on my dataset?

Testing a new dataset on this framework should not be too difficult. Do the following (in the order you feel most comfortable):
- define the number of individuals, **n_individuals_NEW**, that your dataset contains, this can be done in the [data_utils.py](https://github.com/DCalhas/eeg_to_fmri/blob/0c634384faa79c7f7289aa7ec1af9b04dac92ebc/src/utils/data_utils.py#L32) file;
- additionally you may define new variables in the [data_utils.py](https://github.com/DCalhas/eeg_to_fmri/blob/0c634384faa79c7f7289aa7ec1af9b04dac92ebc/src/utils/data_utils.py) file, corresponding to **n_individuals_train_NEW** and **n_individuals_test_NEW**, which refer to the number of individuals used for the training and testing set, respectively;
- define **dataset_NEW** variable in the [fmri_utils.py](https://github.com/DCalhas/eeg_to_fmri/blob/0c634384faa79c7f7289aa7ec1af9b04dac92ebc/src/utils/eeg_utils.py#L47) and [eeg_utils.py](https://github.com/DCalhas/eeg_to_fmri/blob/0c634384faa79c7f7289aa7ec1af9b04dac92ebc/src/utils/fmri_utils.py#L43) files. At this point you might be thinking: "Why is this guy defining the same variable in two different places?", well he ain't too smart tbh and he lazy af;
- define the frequency, **fs_NEW**, at which the EEG recording was sampled, this can be done in the [eeg_utils.py](https://github.com/DCalhas/eeg_to_fmri/blob/0c634384faa79c7f7289aa7ec1af9b04dac92ebc/src/utils/eeg_utils.py#L38) file;
- define the Time Response, **TR_NEW**, at which each fMRI volume was sampled, this can be done in the [fmri_utils.py](https://github.com/DCalhas/eeg_to_fmri/blob/0c634384faa79c7f7289aa7ec1af9b04dac92ebc/src/utils/fmri_utils.py#L27);
- additionally, you might want to define the list of channels (if your EEG electrode setup follows the [10-20 system](https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG))), to retrieve more advanced analysis, such as EEG electrode relevance. This should be done in the beginning of the [eeg_utils.py](https://github.com/DCalhas/eeg_to_fmri/blob/0c634384faa79c7f7289aa7ec1af9b04dac92ebc/src/utils/eeg_utils.py) file;
- last, but no least, comes the time to implement the two functions that read the EEG and fMRI recordings, corresponding to **get_eeg_instance_NEW**, at [eeg_utils.py](https://github.com/DCalhas/eeg_to_fmri/blob/0c634384faa79c7f7289aa7ec1af9b04dac92ebc/src/utils/eeg_utils.py#L171), and **get_indviduals_path_NEW**, at [fmri_utils.py](https://github.com/DCalhas/eeg_to_fmri/blob/0c634384faa79c7f7289aa7ec1af9b04dac92ebc/src/utils/fmri_utils.py#L299);

In addition to reading the rest of this section, which helps you setting up your data, you also have available two blog posts:

- [EEG recording to fMRI volume](https://dcalhas.github.io/eeg_to_fmri/blog/EEG_fMRI.html): goes over an example on how to operate with a simultaneous EEG and fMRI dataset and creates a model that synthesizes fMRI from EEG;
- [Classification on EEG only datasets](https://dcalhas.github.io/eeg_to_fmri/blog/Sinusoid_separation.html): this one picks up on the previous blog post and uses the pretrained model (that synthesizes fMRI from EEG), and shows you how to create an fMRI view of an EEG instance and classify it.

### Dataset structure

In this example, we assume your dataset has the following structure (if it has a different structure please interpret the code provided in the next two sections and adapt it):

```
NEW
|	.
|	..
|	README.md
└───────EEG
|	└───────sub-001
|	|	|	FILE.eeg
|	|	|	FILE.vhdr
|	|	|	FILE.vmrk
|	└───────sub-002
|	...
└───────BOLD
	└───────sub-001
	|	|	FILE.anat
	|	|	FILE.nii.gz
	└───────sub-002
	...
```

#### Implementing the get_eeg_instance_NEW function

Ideally you want this function to return an [mne.io.Raw](https://mne.tools/stable/generated/mne.io.Raw.html) object, that contains the EEG data. In this "tutorial" only this is the only supported option, however do it as you like most.

The inputs of this function are:
- *individual* - int, the individual one wants to retrieve. This function is being executed inside a for loop, ```for individual in range(getattr(data_utils, "n_individuals_"+dataset)```, that goes through the range of individuals, **n_individuals_NEW**, you set in the [data_utils.py](https://github.com/DCalhas/eeg_to_fmri/blob/0c634384faa79c7f7289aa7ec1af9b04dac92ebc/src/utils/data_utils.py#L32) file;
- *path_eeg* - str, the path where your dataset is located, e.g. ```path_eeg=os.environ['EEG_FMRI_DATASETS']+dataset_NEW+"/EEG/"```, this may be an optional argument set as ```path_eeg=os.environ['EEG_FMRI_DATASETS']+dataset_NEW+"/EEG/"```;
- *task* - str, can be set to None if it does not apply to your dataset;

So given these inputs one can start by listing the directories of your dataset (now this can depend on how you organized the data, we assume that each individual has a folder dedicated to itself and the sorted names of the folders have the individual's folders first and after the auxiliary description ones, e.g. "info" for information about the dataset):


```python
def get_eeg_instance_NEW(individual, path_eeg=os.environ['EEG_FMRI_DATASETS']+dataset_NEW+"/EEG/", task=None,):
	individuals = sorted([f for f in listdir(path_eeg) if isdir(join(path_eeg, f))])

	individual = individuals[individual]

	path=path_eeg+individual+"/"
	print(path)#for debug purposes only, please remove this line after function is implemented
```
The output of the last print (if ```individuals=["sub-001", "sub-002", ..., "sub-"+data_utils.n_individuals_NEW, ...]```):
```bash
/tmp/"+dataset_NEW+"/EEG/sub-001/
```

Inside the path described above should be a set of files needed to load a eeg brainvision object. If you sort these files, likely 
the ```.vhdr``` is the second option:

```python
	brainvision_files = sorted([f for f in listdir(path) if isfile(join(path, f))])
	vhdr_file = brainvision_files[1]
```

Now one only needs to return the Brainvision object:

```python
	complete_path = path + vhdr_file
	return mne.io.read_raw_brainvision(complete_path, preload=True, verbose=0)
```

In the end **get_eeg_instance_NEW** is:

```python
def get_eeg_instance_NEW(individual, path_eeg=os.environ['EEG_FMRI_DATASETS']+dataset_NEW+"/", task=None,):
	individuals = sorted([f for f in listdir(path_eeg) if isdir(join(path_eeg, f))])

	individual = individuals[individual]

	path=path_eeg+individual+"/"
	
	brainvision_files = sorted([f for f in listdir(path) if isfile(join(path, f))])
	vhdr_file = brainvision_files[1]

	complete_path = path + vhdr_file
	return mne.io.read_raw_brainvision(complete_path, preload=True, verbose=0)
```

#### Implementing the get_individuals_path_NEW function

Next step is to implement the function that retrieves the fMRI recordings of all individuals. We assume each individual's recording is save in an [.nii.gz](http://justsolve.archiveteam.org/wiki/NII) file.

The inputs of this function are:
- *path_fmri* - str, absolute path that specifies the location of your dataset, e.g. ```path_fmri=os.environ['EEG_FMRI_DATASETS']+dataset_NEW+"/BOLD/```, this may be an optional argument set as ```path_fmri=os.environ['EEG_FMRI_DATASETS']+dataset_NEW+"/BOLD/"```
- *resolution_factor* - float, this is an optional argument that might not be used, please refer to the [functions](https://github.com/DCalhas/eeg_to_fmri/blob/1df6f6e353952ca6b9643938e1558ecf0697d435/src/utils/fmri_utils.py#L110) where this argument is used to grasp its function. WARNING: this variable is deprecated;
- *number_individuals* - int, this variables specifies the number of individuals in this dataset, it is specified in the function call as ```number_individuals=getattr(data_utils, "number_individuals_"+dataset)```;

Given the absolute path of the data and the number of individuals one wants to retrieve, we can now start implementing the code. Let's start by listing the individuals and saving it in a list:

```python
def get_individuals_paths_NEW(path_fmri=os.environ['EEG_FMRI_DATASETS']+dataset_NEW+"/BOLD/", resolution_factor=None, number_individuals=None):
	fmri_individuals = []#this will be the output of this function
	
	dir_individuals = sorted([f for f in listdir(path_fmri) if isdir(join(path_fmri, f)) and "sub" in path_fmri+f])
	print(dir_individuals)
```

```bash
[os.environ['EEG_FMRI_DATASETS']+dataset_NEW+"/BOLD/sub-001", os.environ['EEG_FMRI_DATASETS']+dataset_NEW+"/BOLD/sub-002", ..., os.environ['EEG_FMRI_DATASETS']+dataset_NEW+"/BOLD/sub-"+data_utils.n_individuals_NEW, ...]
```

Now we can move on to start the loop, where one iterates over each individuals' directory and loads the recording:

```
	for i in range(number_individuals):
		task_file=sorted([f for f in listdir(path_fmri+dir_individuals[i]) if isfile(path_fmri+dir_individuals[i]+f) and task in path_fmri+dir_individuals[i]+f])

		print(task_file)
```

We assume that inside the individuals' folder, you will have an ".nii.gz" file and an additional ".anat" file. When sorted this list will have the ".nii.gz" file in the second place:

```
[os.environ['EEG_FMRI_DATASETS']+dataset_NEW+"/BOLD/sub-001/FILE.anat", os.environ['EEG_FMRI_DATASETS']+dataset_NEW+"/BOLD/sub-001/FILE.nii.gz"]
```

Therefore we pick the second file and use the [nilearn](https://nilearn.github.io/modules/generated/nilearn.image.load_img.html) library to load the image:

```python
		file_path= path_fmri+dir_individuals[i]+task_file[1]
		
		fmri_individuals += [image.load_img(file_path)]

	return fmri_individuals
```

In the end, this function is as:

```python
def get_individuals_paths_NEW(path_fmri=os.environ['EEG_FMRI_DATASETS']+dataset_NEW+"/BOLD/", resolution_factor=None, number_individuals=None):
	fmri_individuals = []#this will be the output of this function
	
	dir_individuals = sorted([f for f in listdir(path_fmri) if isdir(join(path_fmri, f)) and "sub" in path_fmri+f])
	
	for i in range(number_individuals):
		task_file=sorted([f for f in listdir(path_fmri+dir_individuals[i]) if isfile(path_fmri+dir_individuals[i]+f) and task in path_fmri+dir_individuals[i]+f])

		file_path= path_fmri+dir_individuals[i]+task_file[1]
		
		fmri_individuals += [image.load_img(file_path)]

	return fmri_individuals
```

##### My dataset has fMRI volumes with higher resolutions than accepted by this work. What should I do?

Unfortunately, the model only accepts:
- EEG instances with 64 channels and a total of 134 frequency resolutions in a specified window of **TR_\***;
- fMRI instances with 64x64x30 resolution.

For the EEG, we do not have a specified studied solution, just pray that it works.

For the fMRI, we found that mutating the resolution via Discrete Cosine Transform (DCT) spectral domain is a reasonable work around. To do this you only need to add the specified lines to the **get_individuals_paths_NEW** and a *downsample* and *downsample_shape* optional arguments to the function call:

```python
def get_individuals_paths_NEW(path_fmri=os.environ['EEG_FMRI_DATASETS']+dataset_NEW+"/BOLD/", resolution_factor=None, number_individuals=None, downsample=True, downsample_shape=(64,64,30)):
	
	...

	if(downsample):
		import sys
		sys.path.append("..")
		from layers import fft
		dct=None
		idct=None

	...
```

Import the modules to perform the DCT and either add or remove resolutions to fit your data to the desired shapes. Then when you load the image, you should mutate it as:

```python
	
		...
		fmri_individuals += [image.load_img(file_path)]

		if(downsample):
			img = np.swapaxes(np.swapaxes(np.swapaxes(fmri_individuals[-1].get_fdata(), 0, 3), 1,2), 1,3)
			if(dct is None and idct is None):
				dct = fft.DCT3D(*img.shape[1:])
				idct = fft.iDCT3D(*downsample_shape)
			fmri_individuals[-1] = image.new_img_like(fmri_individuals[-1], 
														np.swapaxes(np.swapaxes(np.swapaxes(idct(dct(img).numpy()[:, :downsample_shape[0], :downsample_shape[1], :downsample_shape[2]]).numpy(), 0, 3), 0,2), 0,1))
		return fmri_individuals
```

After this, you should be set to run the code and retrieve the results you desire.

### Run the code and retrieve results

Now you just need to run the [main.py](https://github.com/DCalhas/eeg_to_fmri/blob/master/src/main.py) file with your dataset identifier given as an argument. Please refer to the [documentation](https://github.com/DCalhas/eeg_to_fmri/blob/master/DOCUMENTATION.md), where you will find what you need to give as arguments, an example call is (open shell):

```shell
cd eeg_to_fmri/src
conda activate eeg_fmri
mkdir /tmp/eeg_to_fmri
mkdir /tmp/eeg_to_fmri/metrics
python main.py metrics NEW -na_path_eeg ../na_models_eeg/na_specification_2 -na_path_fmri ../na_models_fmri/na_specification_2 -save_metrics -metrics_path /tmp/eeg_to_fmri/metrics
```

## Blog posts

This repository goes along with blog posts done during my PhD course:

- [EEG recording to fMRI volume](https://dcalhas.github.io/eeg_to_fmri/blog/EEG_fMRI.html);
- [Classification on EEG only datasets](https://dcalhas.github.io/eeg_to_fmri/blog/Sinusoid_separation.html);

## Acknowledgements

We would like to thank everyone at [INESC-ID](https://www.inesc-id.pt/) that accompanied the journey throughout my PhD. This work was supported by national funds through Fundação para a Ciência e Tecnologia ([FCT](https://www.fct.pt/index.phtml.pt)), under the Ph.D. Grant SFRH/BD/5762/2020 to David Calhas and INESC-ID pluriannual UIDB/50021/2020.

## Cite this repository

If you use this software in your work, please cite it using the following metadata:

```
@article{calhas2022eeg,
  title={EEG to fMRI Synthesis Benefits from Attentional Graphs of Electrode Relationships},
  author={Calhas, David and Henriques, Rui},
  journal={arXiv preprint arXiv:2203.03481},
  year={2022}
}
```


## License

[MIT License](https://choosealicense.com/licenses/mit/)
