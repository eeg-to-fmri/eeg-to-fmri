from multiprocessing import Process

import os

MAX_NUMBER_ALLOC=1000

def process_setup_tensorflow(memory_limit, seed=42, run_eagerly=False):
	from utils import tf_config

	tf_config.set_seed(seed=seed)
	tf_config.setup_tensorflow(device="GPU", memory_limit=memory_limit, run_eagerly=run_eagerly)

def launch_process(function, args):
	p = Process(target=function, args=args)
	p.daemon = True
	p.start()
	p.join()


def theta_latent_fmri():
	return [{'name': 'learning_rate', 'type': 'continuous',
			'domain': (1e-10, 1e-2)},
			{'name': 'weight_decay', 'type': 'continuous',
			'domain': (1e-10, 1e-1)},
			{'name': 'batch_size', 'type': 'discrete',
			'domain': (64,)},
			{'name': 'latent', 'type': 'discrete',
			'domain': (4,5,6,7,8,9,10,15,20)},
			{'name': 'channels', 'type': 'discrete',
			'domain': (2,4)},
			{'name': 'max_pool', 'type': 'discrete',
			'domain': (0,1)},
			{'name': 'batch_norm', 'type': 'discrete',
			'domain': (0,1)},
			{'name': 'skip', 'type': 'discrete',
			'domain': (0,1)},
			{'name': 'dropout', 'type': 'discrete',
			'domain': (0,1)},
			{'name': 'out_filter', 'type': 'discrete',
			'domain': (0,1,2)}]


def load_data_latent_fmri(dataset, n_individuals, n_individuals_train, n_volumes, interval_eeg, memory_limit):
	from utils import preprocess_data
	import tensorflow as tf

	process_setup_tensorflow(memory_limit)

	with tf.device('/CPU:0'):
		train_data, test_data = preprocess_data.dataset(dataset, n_individuals=n_individuals, 
												interval_eeg=interval_eeg, 
												ind_volume_fit=False,
												standardize_fmri=True,
												iqr=False,
												verbose=True)

	return train_data[1]

def load_data_eeg_fmri(dataset, n_individuals, n_volumes, interval_eeg, memory_limit, return_test=False, setup_tf=True):
	from utils import preprocess_data
	import tensorflow as tf

	if(setup_tf):
		process_setup_tensorflow(memory_limit)

	with tf.device('/CPU:0'):
		train_data, test_data = preprocess_data.dataset(dataset, n_individuals=n_individuals, 
												interval_eeg=interval_eeg, 
												ind_volume_fit=False,
												standardize_fmri=True,
												iqr=False,
												verbose=True)
	if(return_test):
		return train_data, test_data
	return train_data


def make_dir_batches(dataset, n_individuals, n_individuals_train, n_individuals_val, n_volumes, interval_eeg, memory_limit, batch_size, batch_path):
	import tensorflow as tf
	import os
	import numpy as np
	import shutil
	from pathlib import Path

	dataset = load_data_eeg_fmri(dataset, n_individuals, n_volumes, interval_eeg, memory_limit)

	eeg, fmri = dataset

	#partition in train and validation sets
	eeg_val = eeg[n_individuals_train*n_volumes:(n_individuals_train+n_individuals_val)*n_volumes]
	eeg_train = eeg[:n_individuals_train*n_volumes]

	fmri_val = fmri[n_individuals_train*n_volumes:(n_individuals_train+n_individuals_val)*n_volumes]
	fmri_train = fmri[:n_individuals_train*n_volumes]


	dataset = tf.data.Dataset.from_tensor_slices((eeg_train, fmri_train)).batch(batch_size)

	if(Path(batch_path).exists()):
		shutil.rmtree(batch_path)

	#save
	os.mkdir(batch_path)

	batch=1
	#write train batches
	for batch_x, batch_y in dataset.repeat(1):
		np.save(batch_path+"/batch_x_"+str(batch), batch_x.numpy())
		np.save(batch_path+"/batch_y_"+str(batch), batch_y.numpy())
		batch+=1


def load_batch(tensorflow, batch_path, batch, dtype):
	import numpy as np

	batch_x, batch_y = (np.load(batch_path+"/batch_x_"+str(batch)+".npy"),
						np.load(batch_path+"/batch_y_"+str(batch)+".npy"))
	return (tensorflow.convert_to_tensor(batch_x, dtype=dtype), 
			tensorflow.convert_to_tensor(batch_y, dtype=dtype))


def batch_prediction(shared_flattened_predictions, setup, batch_path, batch, epoch, network, na_path, batch_size, learning_rate, memory_limit, best_eeg, seed):
	#imports
	import tensorflow as tf
	from utils import train, losses_utils, state_utils, tf_config
	from layers.fourier_features import RandomFourierFeatures
	from models.eeg_to_fmri import EEG_to_fMRI, call
	from models.fmri_ae import fMRI_AE
	import pickle

	tf_config.setup_tensorflow(memory_limit=memory_limit, device="GPU")
	tf.random.set_seed(seed)

	#load batch
	eeg, fmri = load_batch(tf, batch_path, batch, tf.float32)

	#unroll hyperparameters
	theta = (0.002980911194116198, 0.0004396489214334123, (9, 9, 4), (1, 1, 1), 4, (7, 7, 7), 4, True, True, True, True, 3, 1)
	learning_rate=float(theta[0])
	weight_decay = float(theta[1])
	kernel_size = theta[2]
	stride_size = theta[3]
	batch_size=int(theta[4])
	latent_dimension=theta[5]
	n_channels=int(theta[6])
	max_pool=bool(theta[7])
	batch_norm=bool(theta[8])
	skip_connections=bool(theta[9])
	dropout=bool(theta[10])
	n_stacks=int(theta[11])
	outfilter=int(theta[12])
	local=True
	
	if(setup=="fmri"):
		with open(best_eeg, "rb") as f:
			na_specification_eeg = pickle.load(f)
		with open(na_path + "/na_specification_"+str(network+1), "rb") as f:
			na_specification_fmri = pickle.load(f)
	elif(setup=="eeg"):
		with open(na_path + "/na_specification_"+str(network+1), "rb") as f:
			na_specification_eeg = pickle.load(f)
	else:
		raise NotImplementedError

	#load or build model
	with tf.device('/CPU:0'):
		loss_fn = losses_utils.mse_cosine

		if(batch == 1 and epoch == 0):
			#TODO: correct me to load right model specification
			optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
			#kernels, strides = parse_conv.cnn_to_tuple(tf.keras.models.load_model(na_path + method + "/architecture_" + str(network+1), compile=False))
			model = EEG_to_fMRI(latent_dimension, eeg.shape[1:], na_specification_eeg, 4, weight_decay=0.000, skip_connections=True,
											fourier_features=True, random_fourier=True, topographical_attention=False,
											batch_norm=True, local=True, seed=None, fmri_args = (latent_dimension, fmri.shape[1:], 
											kernel_size, stride_size, n_channels, max_pool, batch_norm, weight_decay, skip_connections,
											n_stacks, True, False, outfilter, dropout, None, False, na_specification_fmri))
			model.build(eeg.shape, fmri.shape)
			model.compile(optimizer=optimizer)
		else:
			#load model and optimizer at previous state
			model = tf.keras.models.load_model(na_path + "/architecture_" + str(network) + "_training", compile=True, 
										custom_objects={"EEG_to_fMRI": EEG_to_fMRI,
														"fMRI_AE": fMRI_AE,
														"RandomFourierFeatures": RandomFourierFeatures})
			state_utils.setup_state(tf, model.optimizer, na_path  + "/architecture_" + str(network) + "_training/opt_config", 
												na_path + "/architecture_" + str(network) + "_training/gen_config")

	loss, batch_preds = train.train_step(model, (eeg, fmri), model.optimizer, loss_fn, u_architecture=True, return_logits=True, call_fn=call)
	loss=loss.numpy()

	flattened_batch_preds=batch_preds.numpy().flatten()
	for i in range(flattened_batch_preds.shape[0]):
		shared_flattened_predictions[i] = flattened_batch_preds[i]

	print("NA", network, " at epoch", epoch+1, " and batch", batch, "with loss:", loss, end="\n")
	
	#save model
	model.save(na_path + "/architecture_" + str(network) + "_training", save_format="tf", save_traces=False)
	#save state
	state_utils.save_state(tf, model.optimizer, na_path + "/architecture_" + str(network) + "_training/opt_config", 
							na_path + "/architecture_" + str(network) + "_training/gen_config")

def continuous_training(o_predictions, batch_path, batch, learning_rate, epoch, na_path, gpu_mem, seed):
	import tensorflow as tf
	import numpy as np
	from utils import state_utils, losses_utils, tf_config, train
	from models import softmax
	
	tf_config.setup_tensorflow(memory_limit=gpu_mem, device="GPU")
	tf.random.set_seed(seed)

	loss_fn = losses_utils.mse

	if(batch==1 and epoch == 0):
		opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
		model=softmax.Softmax((o_predictions.shape[0],))
		model.build(input_shape=o_predictions.shape)
		model.compile(optimizer=opt)
	else:
		model=tf.keras.models.load_model(na_path + "/softmax_training", compile=True)
		#opt = state_utils.setup_state(tf, opt, na_path + method + "/softmax_training/opt_config", 
		state_utils.setup_state(tf, model.optimizer, na_path + "/softmax_training/opt_config", 
							na_path + "/softmax_training/gen_config")
	
	#load batch
	_, fmri = load_batch(tf, batch_path, batch, tf.float32)

	#training step to update weights with batch
	loss = train.train_step(model, (o_predictions,fmri), model.optimizer, loss_fn).numpy()
	
	print("Softmax epoch loss: ", loss, end="\n\n\n")
	print(model.trainable_variables[0].numpy())

	model.save(na_path + "/softmax_training", save_format="tf")
	#save state
	state_utils.save_state(tf, model.optimizer, na_path + "/softmax_training/opt_config", 
							na_path + "/softmax_training/gen_config")



def save_weights(epoch, na_path, save_weights_path): 
	import tensorflow as tf
	import numpy as np

	model=tf.keras.models.load_model(na_path + "/softmax_training", compile=False)

	np.save(save_weights_path+"/epoch_"+str(epoch), model.trainable_variables[0].numpy())


def cross_validation_latent_fmri(score, learning_rate, weight_decay, 
						kernel_size, stride_size,
						batch_size, latent_dimension, n_channels, 
						max_pool, batch_norm, skip_connections, dropout,
						n_stacks, outfilter, dataset, n_individuals, 
						n_individuals_train, n_volumes, 
						interval_eeg, memory_limit):

	from utils import train
	from models import fmri_ae
	from sklearn.model_selection import KFold
	import tensorflow as tf

	data = load_data_latent_fmri(dataset, n_individuals, n_individuals_train, n_volumes, interval_eeg, memory_limit)
	n_folds = 5

	for train_idx, val_idx in KFold(n_folds).split(data):
		with tf.device('/CPU:0'):
			x_train = data[train_idx]
			x_val = data[val_idx]
			
			#build model
			model = fmri_ae.fMRI_AE(latent_dimension, x_train.shape[1:], kernel_size, stride_size, n_channels,
								maxpool=max_pool, batch_norm=batch_norm, weight_decay=weight_decay, skip_connections=skip_connections,
								n_stacks=n_stacks, local=True, local_attention=False, outfilter=outfilter, dropout=dropout)
			model.build(input_shape=x_train.shape)

			#train model
			optimizer = tf.keras.optimizers.Adam(learning_rate)
			loss_fn = tf.keras.losses.MSE#replace

			train_set = tf.data.Dataset.from_tensor_slices((x_train, x_train)).batch(batch_size)
			dev_set = tf.data.Dataset.from_tensor_slices((x_val, x_val)).batch(1)
		
		train.train(train_set, model, optimizer, 
								loss_fn, epochs=10, 
								val_set=None, verbose=True)

		#evaluate
		score.value += train.evaluate(dev_set, model, loss_fn)

	score.value = (score.value-1.0)/n_folds


def cross_validation_eeg_fmri(score, fourier_features, random_fourier,
						topographical_attention, conditional_attention_style, 
						na_specification_eeg, na_specification_fmri, 
						learning_rate, weight_decay, 
						kernel_size, stride_size,
						batch_size, latent_dimension, n_channels, 
						max_pool, batch_norm, skip_connections, dropout,
						n_stacks, outfilter, dataset, n_individuals, 
						n_individuals_train, n_volumes, 
						interval_eeg, memory_limit):

	from utils import train, losses_utils
	from models import eeg_to_fmri
	from sklearn.model_selection import KFold
	import tensorflow as tf

	data = load_data_eeg_fmri(dataset, n_individuals, n_volumes, interval_eeg, memory_limit, return_test=False, setup_tf=True)
	n_folds = 5

	for train_idx, val_idx in KFold(n_folds).split(data[0]):
		with tf.device('/CPU:0'):
			x_train, y_train = (data[0][train_idx], data[1][train_idx])
			x_val, y_val = (data[0][val_idx], data[1][val_idx])
			
			model = eeg_to_fmri.EEG_to_fMRI(latent_dimension, x_train.shape[1:], na_specification_eeg, n_channels,
								weight_decay=weight_decay, skip_connections=True,
								batch_norm=True, #dropout=False,
								fourier_features=fourier_features,
								random_fourier=random_fourier,
								topographical_attention=topographical_attention,
								conditional_attention_style=conditional_attention_style,
								inverse_DFT=False, DFT=False,
								variational_iDFT=False,
								variational_coefs=(15,15,15),
								low_resolution_decoder=False,
								local=True, seed=None, 
								fmri_args = (latent_dimension, y_train.shape[1:], 
								kernel_size, stride_size, n_channels, 
								max_pool, batch_norm, weight_decay, skip_connections,
								n_stacks, True, False, outfilter, dropout, None, False, na_specification_fmri))
			
			model.build(x_train[0].shape, x_train[1].shape)
			
			optimizer = tf.keras.optimizers.Adam(learning_rate)
			loss_fn = losses_utils.mae_cosine

			train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
			dev_set = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(1)


		train.train(train_set, model, optimizer, 
								loss_fn, epochs=10, 
								val_set=None, verbose=True)

		#evaluate
		score.value += train.evaluate(dev_set, model, loss_fn)

	score.value = (score.value-1.0)/n_folds


def train_synthesis(dataset, epochs, style_prior, padded, variational, variational_coefs, variational_dependent_h, variational_dist, variational_random_padding, resolution_decoder, aleatoric_uncertainty, save_path, gpu_mem, seed, run_eagerly):
	#imports
	import tensorflow as tf

	from utils import data_utils, preprocess_data, tf_config, train, losses_utils

	from models import eeg_to_fmri

	interval_eeg=10
	tf_config.set_seed(seed=seed)#02 20
	tf_config.setup_tensorflow(device="GPU", memory_limit=gpu_mem, run_eagerly=run_eagerly)

	from pathlib import Path

	import numpy as np

	import pickle

	raw_eeg=False

	theta = (0.002980911194116198, 0.0004396489214334123, (9, 9, 4), (1, 1, 1), 4, (7, 7, 7), 4, True, True, True, True, 3, 1)
	#unroll hyperparameters
	learning_rate=float(theta[0])
	weight_decay = float(theta[1])
	kernel_size = theta[2]
	stride_size = theta[3]
	batch_size=int(theta[4])
	latent_dimension=theta[5]
	n_channels=int(theta[6])
	max_pool=bool(theta[7])
	batch_norm=bool(theta[8])
	skip_connections=bool(theta[9])
	dropout=bool(theta[10])
	n_stacks=int(theta[11])
	outfilter=int(theta[12])
	local=True
	with open(os.environ['EEG_FMRI']+"/na_models_eeg/na_specification_2", "rb") as f:
		na_specification_eeg = pickle.load(f)
	with open(os.environ['EEG_FMRI']+"/na_models_fmri/na_specification_2", "rb") as f:
		na_specification_fmri = pickle.load(f)

	with tf.device('/CPU:0'):
		
		train_data, _ = preprocess_data.dataset(dataset, 
												n_individuals=getattr(data_utils, "n_individuals_"+dataset),
												interval_eeg=interval_eeg, 
												ind_volume_fit=False,
												standardize_fmri=True,
												iqr=False,
												verbose=True)
		eeg_train, fmri_train = train_data
		
		#placeholder not pretty please correct me
		_resolution_decoder=None
		if(type(resolution_decoder) is float):
			_resolution_decoder=(int(fmri_train.shape[1]/resolution_decoder),int(fmri_train.shape[2]/resolution_decoder),int(fmri_train.shape[3]/resolution_decoder))
		model = eeg_to_fmri.EEG_to_fMRI(latent_dimension, eeg_train.shape[1:], na_specification_eeg, n_channels,
							weight_decay=weight_decay, skip_connections=True, batch_norm=True, fourier_features=True, random_fourier=True,
							topographical_attention=True, conditional_attention_style=True, conditional_attention_style_prior=style_prior,
							inverse_DFT=variational or padded, DFT=variational or padded, variational_iDFT=variational, variational_coefs=variational_coefs, 
							variational_iDFT_dependent=variational_dependent_h>1, variational_iDFT_dependent_dim=variational_dependent_h, aleatoric_uncertainty=aleatoric_uncertainty, 
							low_resolution_decoder=type(resolution_decoder) is float, variational_random_padding=variational_random_padding, 
							resolution_decoder=_resolution_decoder, local=True, seed=None, 
							fmri_args = (latent_dimension, fmri_train.shape[1:], 
							kernel_size, stride_size, n_channels, 
							max_pool, batch_norm, weight_decay, skip_connections,
							n_stacks, True, False, outfilter, dropout, None, False, na_specification_fmri))
		model.build(eeg_train.shape, fmri_train.shape)
		optimizer = tf.keras.optimizers.Adam(learning_rate)
		loss_fn = list(losses_utils.LOSS_FNS.values())[int(aleatoric_uncertainty)]#if variational get loss fn at index 1
		train_set = tf.data.Dataset.from_tensor_slices((eeg_train, fmri_train)).batch(batch_size)

	print("I: Starting pretraining of synthesis network")

	loss_history = train.train(train_set, model, optimizer, 
							loss_fn, epochs=epochs, 
							u_architecture=True,
							val_set=None, verbose=True, verbose_batch=False)[0]

	print("I: Saving synthesis network at", save_path)

	model.save(save_path, save_format="tf", save_traces=False)

def create_labels(view, dataset, path, setting):

	import numpy as np

	y_pred = np.empty((0,), dtype="float32")
	y_true = np.empty((0,), dtype="float32")

	np.save(path+setting+"/y_pred.npy", y_pred, allow_pickle=True)
	np.save(path+setting+"/y_true.npy", y_true, allow_pickle=True)

def append_labels(view, path, y_true, y_pred, setting):
	import numpy as np
	np.save(path+setting+"/y_pred.npy",np.append(np.load(path+setting+"/y_pred.npy", allow_pickle=True), y_pred), allow_pickle=True)
	np.save(path+setting+"/y_true.npy",np.append(np.load(path+setting+"/y_true.npy", allow_pickle=True), y_true), allow_pickle=True)


def setup_data_loocv(setting, view, dataset, fold, n_folds_cv, n_processes, epochs, gpu_mem, seed, run_eagerly, save_explainability, path_network, path_labels, feature_selection=False, segmentation_mask=False, style_prior=False, variational=False):

	from utils import preprocess_data

	from multiprocessing import Manager

	launch_process(load_data_loocv,
					(view, dataset, path_labels))

	dataset_clf_wrapper = preprocess_data.Dataset_CLF_CV(dataset, standardize_eeg=True, load=False, load_path=path_labels)

	for i in range(fold, dataset_clf_wrapper.n_individuals):
		#CV hyperparameter l1 and l2 reg constants
		hyperparameters = cv_opt(i, n_processes, n_folds_cv, view, dataset, epochs, gpu_mem, seed, run_eagerly, path_labels, path_network, feature_selection=feature_selection, segmentation_mask=segmentation_mask, variational=variational)

		#validate
		launch_process(loocv,
					(i, setting, view, dataset, hyperparameters[0], epochs, hyperparameters[2], hyperparameters[1], n_processes*(gpu_mem), seed, run_eagerly, save_explainability, path_network, path_labels, feature_selection, segmentation_mask, style_prior, variational))
def load_data_loocv(view, dataset, path_labels):
	from utils import preprocess_data
	
	dataset_clf_wrapper = preprocess_data.Dataset_CLF_CV(dataset,
														eeg_limit=True, raw_eeg=view=="raw",
														eeg_f_limit=135, standardize_eeg=True, 
														load=True, load_path=None)

	dataset_clf_wrapper.save(path_labels)

def predict(test_set, model):
	import numpy as np
	import tensorflow as tf

	hits = np.empty((0,))
	y_true = np.empty((0,))
	y_pred = np.empty((0,))

	for x,y in test_set.repeat(1):
		if(tf.math.reduce_all(tf.math.equal(tf.cast(y==1.0,tf.int64), tf.cast(tf.keras.activations.sigmoid(model(x))[0]>=0.5, tf.int64)).numpy())):
			hits = np.append(hits, 1.0)
		else:
			hits = np.append(hits, 0.0)
			
		if(y.numpy()[0]==1.0):
			y_true=np.append(y_true,1.0)
		else:
			y_true=np.append(y_true,0.0)
		
		y_pred=np.append(y_pred, tf.keras.activations.sigmoid(model(x)).numpy()[0,0])
	
	return hits, y_true, y_pred


def views(model, test_set, y):
	from utils import fmri_utils
	import tensorflow as tf
	import numpy as np

	dev_views = np.empty((0,)+getattr(fmri_utils, "fmri_shape_01")+(1,))
	for x, _ in test_set.repeat(1):
		dev_views = np.append(dev_views, model.view(x)[0], axis=0)
	
	return tf.data.Dataset.from_tensor_slices((dev_views,y)).batch(1)


def cv_opt(fold_loocv, n_processes, n_folds_cv, view, dataset, epochs, gpu_mem, seed, run_eagerly, path_labels, path_network, feature_selection=False, segmentation_mask=False, variational=False):
	import GPyOpt
	
	iteration=0

	def filter_shared_array(sh_array):
		import numpy as np

		x = np.empty((0,), dtype=np.float32)
		for i in range(len(sh_array)):
			if(str(sh_array[i])=='nan'):
				return x
			x = np.append(x, [sh_array[i]], axis=0)

		return x

	def optimize_wrapper(theta):
		from multiprocessing import Manager

		l1_reg, batch_size, learning_rate = (float(theta[:,0]), int(theta[:,1]), float(theta[:,2]))
		if(n_processes==1):
			value = Manager().Array('d', range(1))
			launch_process(optimize_elastic, (value, (l1_reg, batch_size, learning_rate),))
		elif(n_processes>1):
			value=[optimize_elastic_multi_process((l1_reg, batch_size, learning_rate))]

		print("Finished with score", value[0], end="\n\n\n")
		return value[0]

	def optimize_elastic_multi_process(theta):

		def run_fold(theta, fold):
			from utils import preprocess_data, tf_config, train, losses_utils, metrics
			from models import eeg_to_fmri, classifiers
			import tensorflow as tf
			import os
			import numpy as np
			from sklearn.utils import shuffle
			import gc

			l2_reg, batch_size, learning_rate = (theta)

			tf_config.set_seed(seed=seed)
			tf_config.setup_tensorflow(device="GPU", memory_limit=gpu_mem, run_eagerly=run_eagerly, set_primary_memory=True, set_tf_threads=True)

			dataset_clf_wrapper = preprocess_data.Dataset_CLF_CV(dataset, standardize_eeg=True, load=False, load_path=path_labels, seed=seed)
			train_data, test_data = dataset_clf_wrapper.split(fold_loocv)
			dataset_clf_wrapper.X = train_data[0]
			dataset_clf_wrapper.y = train_data[1]
			dataset_clf_wrapper.shuffle()
			dataset_clf_wrapper.set_folds(n_folds_cv)

			y_true=np.empty((0,), dtype=np.float32)
			y_pred=np.empty((0,), dtype=np.float32)

			train_data, test_data = dataset_clf_wrapper.split(fold)
			X_train, y_train=train_data
			X_test, y_test=test_data
			with tf.device('/CPU:0'):
				optimizer = tf.keras.optimizers.Adam(learning_rate)

				test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test[:,1])).batch(1)
				
				if(view=="fmri"):
					train_set=preprocess_data.DatasetContrastive(X_train, y_train, batch=batch_size, pairs=1, clf=True, seed=seed)
					loss_fn=losses_utils.ContrastiveClassificationLoss(m=np.pi, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
					linearCLF = classifiers.ViewLatentContrastiveClassifier(tf.keras.models.load_model(path_network, custom_objects=eeg_to_fmri.custom_objects), 
																		X_train.shape[1:], activation=tf.keras.activations.linear, #.linear
																		regularizer=tf.keras.regularizers.L1(l=l2_reg), variational=variational,
																		feature_selection=False, segmentation_mask=False, siamese_projection=False,)
				else:
					#the indexation [:,1] is because we were using softmax instead of sigmoid
					train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train[:,1])).batch(batch_size)

					loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True)
					linearCLF = classifiers.LinearClassifier(regularizer=tf.keras.regularizers.L1(l=l2_reg), variational=variational)
				linearCLF.build(X_train.shape)
			gc.collect()

			train.train(train_set, linearCLF, optimizer, loss_fn, epochs=epochs, val_set=None, u_architecture=False, verbose=False, verbose_batch=False)

			#evaluate
			linearCLF.training=False
			#write to shared array
			save_path=str(Path.home())+"/tmp/"+str(fold)
			np.save(save_path+"_pred.npy",tf.keras.activations.sigmoid(linearCLF(X_test)).numpy()[:,0],allow_pickle=True)
			np.save(save_path+"_true.npy",y_test[:,1],allow_pickle=True)
			print("I: Completed fold", fold, "of", n_folds_cv)

		import numpy as np
		from multiprocessing import Process, Manager
		import gc
		import signal

		if(not os.path.isdir(str(Path.home())+"/tmp")):
			os.mkdir(str(Path.home())+"/tmp")

		active=0
		processes=[]
		for fold in range(n_folds_cv):
			processes+=[Process(target=run_fold, args=(theta, fold))]

		for p in processes:
			p.close()
		for fold in range(n_folds_cv):
			processes[fold]=Process(target=run_fold, args=(theta, fold))
			
		for p in processes:
			if(active<n_processes):
				active+=1
				p.start()
			else:
				for p1 in processes:
					if(p1.is_alive()):
						p1.join(timeout=None)
						active-=1
				active=0
				for p1 in processes:
					if(p1.is_alive()):
						active+=1
				#start another process
				p.start()
				active+=1

		for p in processes:
			if(not p._check_closed()):
				p.join(timeout=None)
			try:
				os.kill(p.pid, signal.SIGKILL)
			except:
				print("I: Could not kill process", p.pid, end=".\n")

		y_pred=np.empty((0,),dtype=np.float32)
		y_true=np.empty((0,),dtype=np.float32)
		for fold in range(n_folds_cv):
			try:
				y_pred=np.append(y_pred,np.load(str(Path.home())+"/tmp/"+str(fold)+"_pred.npy",allow_pickle=True), axis=0)
				y_true=np.append(y_true,np.load(str(Path.home())+"/tmp/"+str(fold)+"_true.npy",allow_pickle=True), axis=0)
			except:
				print("Failed in concurrency.")
				return 1/1e-9
			os.remove(str(Path.home())+"/tmp/"+str(fold)+"_pred.npy")
			os.remove(str(Path.home())+"/tmp/"+str(fold)+"_true.npy")			

		acc = np.mean(((y_pred>=0.5).astype("float32")==y_true).astype("float32"))

		del y_pred, y_true, processes
		gc.collect()

		value=1. - acc
		if(np.isnan(value)):
			value = 1/1e-9
		return value


	def optimize_elastic(value, theta):

		from utils import preprocess_data, tf_config, train, losses_utils, metrics

		from models import eeg_to_fmri, classifiers

		import tensorflow as tf

		import numpy as np

		from sklearn.utils import shuffle

		l2_reg, batch_size, learning_rate = (theta)

		tf_config.set_seed(seed=seed)
		tf_config.setup_tensorflow(device="GPU", memory_limit=gpu_mem, run_eagerly=run_eagerly, set_primary_memory=True, set_tf_threads=True)


		dataset_clf_wrapper = preprocess_data.Dataset_CLF_CV(dataset, standardize_eeg=True, load=False, load_path=path_labels)
		train_data, test_data = dataset_clf_wrapper.split(fold_loocv)
		dataset_clf_wrapper.X = train_data[0]
		dataset_clf_wrapper.y = train_data[1]
		dataset_clf_wrapper.shuffle()
		dataset_clf_wrapper.set_folds(n_folds_cv)

		y_true=np.empty((0,), dtype=np.float32)
		y_pred=np.empty((0,), dtype=np.float32)
		for fold in range(n_folds_cv):
			print("On fold", fold+1, end="\r")
			train_data, test_data = dataset_clf_wrapper.split(fold)
			X_train, y_train=train_data
			X_test, y_test=test_data
			with tf.device('/CPU:0'):
				optimizer = tf.keras.optimizers.Adam(learning_rate)

				test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test[:,1])).batch(1)
				
				if(view=="fmri"):
					train_set=preprocess_data.DatasetContrastive(X_train, y_train, batch=batch_size, pairs=1, clf=True)
					loss_fn=losses_utils.ContrastiveClassificationLoss(m=np.pi, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
					linearCLF = classifiers.ViewLatentContrastiveClassifier(tf.keras.models.load_model(path_network, custom_objects=eeg_to_fmri.custom_objects), 
																		X_train.shape[1:], activation=tf.keras.activations.linear, #.linear
																		regularizer=tf.keras.regularizers.L1(l=l2_reg), variational=variational,
																		feature_selection=False, segmentation_mask=False, siamese_projection=False,)
				else:
					#the indexation [:,1] is because we were using softmax instead of sigmoid
					train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train[:,1])).batch(batch_size)

					loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True)
					linearCLF = classifiers.LinearClassifier(regularizer=tf.keras.regularizers.L1(l=l2_reg), variational=variational)
				linearCLF.build(X_train.shape)

			train.train(train_set, linearCLF, optimizer, loss_fn, epochs=epochs, val_set=None, u_architecture=False, verbose=True, verbose_batch=False)
			#evaluate
			linearCLF.training=False
			#evaluate according to final AUC in validation sets
			y_pred=np.append(y_pred, tf.keras.activations.sigmoid(linearCLF(X_test)).numpy()[:,0])
			y_true=np.append(y_true, y_test[:,1])
			
			print("Fold", fold+1, "with Acc:", np.mean(((y_pred>=0.5).astype("float32")==y_true).astype("float32")))

		acc = np.mean(((y_pred>=0.5).astype("float32")==y_true).astype("float32"))
		value[0]=1. - acc
		if(np.isnan(value[0])):
			value[0] = 1/1e-9

	hyperparameters = [{'name': 'l2', 'type': 'continuous','domain': (1e-5, 2.)}, 
						{'name': 'batch_size', 'type': 'discrete', 'domain': (4,8,16)},
						{'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-3, 1e-1)}]
	optimizer = GPyOpt.methods.BayesianOptimization(f=optimize_wrapper, 
													domain=hyperparameters, 
													model_type="GP_MCMC", 
													acquisition_type="EI_MCMC")
	optimizer.run_optimization(max_iter=20)#increase to 20 for real results

	print("Best value: ", optimizer.fx_opt)
	print("Best hyperparameters: \n", optimizer.x_opt)

	return optimizer.x_opt

def loocv(fold, setting, view, dataset, l2_regularizer, epochs, learning_rate, batch_size, gpu_mem, seed, run_eagerly, save_explainability, path_network, path_labels, feature_selection=False, segmentation_mask=False, style_prior=False, variational=False):
	
	from utils import preprocess_data, tf_config, train, lrp, losses_utils

	from models import eeg_to_fmri, classifiers

	import tensorflow as tf

	import os

	import numpy as np

	from sklearn.utils import shuffle

	tf_config.set_seed(seed=seed)
	tf_config.setup_tensorflow(device="GPU", memory_limit=gpu_mem, run_eagerly=run_eagerly, set_primary_memory=True, set_tf_threads=True)

	dataset_clf_wrapper = preprocess_data.Dataset_CLF_CV(dataset, standardize_eeg=True, load=False, load_path=path_labels)

	train_data, test_data = dataset_clf_wrapper.split(fold)
	X_train, y_train = train_data
	X_test, y_test = test_data
	X_train, y_train = shuffle(X_train, y_train)

	with tf.device('/CPU:0'):
		optimizer = tf.keras.optimizers.Adam(learning_rate)
		
		test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test[:,1])).batch(1)

		if(view=="fmri"):
			train_set=preprocess_data.DatasetContrastive(X_train, y_train, batch=batch_size, pairs=1, clf=True)
			loss_fn=losses_utils.ContrastiveClassificationLoss(m=np.pi, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
			linearCLF = classifiers.ViewLatentContrastiveClassifier(tf.keras.models.load_model(path_network, custom_objects=eeg_to_fmri.custom_objects), 
																		X_train.shape[1:], activation=tf.keras.activations.linear, #.linear
																		regularizer=tf.keras.regularizers.L1(l=l2_regularizer), variational=variational,
																		feature_selection=False, segmentation_mask=False, siamese_projection=False,)
		else:
			train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train[:,1])).batch(batch_size)
			loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True)
			linearCLF = classifiers.LinearClassifier(regularizer=tf.keras.regularizers.L1(l=l2_regularizer), variational=variational)
		linearCLF.build(X_train.shape)

	#train classifier
	train.train(train_set, linearCLF, optimizer, loss_fn, epochs=epochs, val_set=None, u_architecture=False, verbose=True, verbose_batch=False)

	#evaluate
	linearCLF.training=False
	#get predictionsf
	hits, y_true, y_pred = predict(test_set, linearCLF)
	#save predictions
	append_labels(view, path_labels, y_true, y_pred, setting)
	#save views of fmri
	if(view=="fmri"):
		if(fold==0):
			np.save(path_labels+setting+"/views.npy", linearCLF.view(X_test)[0].numpy(), allow_pickle=True)
			if(feature_selection or segmentation_mask):
				np.save(path_labels+setting+"/views_relu_regions.npy", linearCLF.view.decoder(X_test).numpy(), allow_pickle=True)
		else:
			np.save(path_labels+setting+"/views.npy", np.append(np.load(path_labels+setting+"/views.npy", allow_pickle=True), linearCLF.view(X_test)[0].numpy(), axis=0), allow_pickle=True)
			if(feature_selection or segmentation_mask):
				np.save(path_labels+setting+"/views_relu_regions.npy", np.append(np.load(path_labels+setting+"/views_relu_regions.npy", allow_pickle=True), linearCLF.view.decoder(X_test).numpy(), axis=0), allow_pickle=True)
	
	print("Finished fold", fold)

	if(save_explainability and view=="fmri"):
		#explaing features
		#explain to fMRI view
		explainer=lrp.LRP(linearCLF.clf)
		R=lrp.explain(explainer, views(linearCLF, test_set, y_test[:,1]), verbose=True)
		#explain to EEG channels
		if(not style_prior):
			raise NotImplementedError
			explainer=lrp.LRP_EEG(linearCLF.view.q_decoder, attention=True, conditional_attention_style=False)
			attention_scores=lrp.explain(explainer, test_set, eeg=True, eeg_attention=True, fmri=False, verbose=True)
		#save explainability
		if(fold==0):
			np.save(path_labels+setting+"/R.npy", R, allow_pickle=True)
			if(not style_prior):
				raise NotImplementedError
				np.save(path_labels+setting+"/attention_scores.npy", attention_scores, allow_pickle=True)
		else:
			np.save(path_labels+setting+"/R.npy", np.append(np.load(path_labels+setting+"/R.npy", allow_pickle=True), R, axis=0), allow_pickle=True)
			if(not style_prior):
				raise NotImplementedError
				np.save(path_labels+setting+"/attention_scores.npy", np.append(np.load(path_labels+setting+"/attention_scores.npy", allow_pickle=True), attention_scores, axis=0), allow_pickle=True)


def compute_acc_metrics(view, path, setting):

	import numpy as np

	y_pred = np.load(path+setting+"/y_pred.npy", allow_pickle=True)
	y_true = np.load(path+setting+"/y_true.npy", allow_pickle=True)

	#true positive
	tp = len(np.where(y_pred[np.where(y_true==1.0)] >= 0.5)[0])
	#true negative
	tn = len(np.where(y_pred[np.where(y_true==0.0)] < 0.5)[0])
	#false positive
	fp = len(np.where(y_pred[np.where(y_true==0.0)] >= 0.5)[0])
	#false negative
	fn = len(np.where(y_pred[np.where(y_true==1.0)] < 0.5)[0])

	print("Accuracy:", (tn+tp)/(tn+tp+fn+fp))
	print("Sensitivity:", (tp)/(tp+fn))
	print("Specificity:", (tn)/(tn+fp))
	print("F1-score:", (tp)/(tp+0.5*(fp+fn)))