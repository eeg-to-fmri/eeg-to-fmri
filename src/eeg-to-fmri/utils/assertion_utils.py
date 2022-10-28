


def main(opt):
	"""
	return the arguments in the opt variable,

	first it performs the according assertions needed for the script to execute

	"""
	mode=opt.mode
	dataset=opt.dataset
	TRs=opt.TRs
	topographical_attention=opt.topographical_attention
	channel_organization=opt.channel_organization
	padded=opt.padded
	variational=opt.variational
	variational_coefs=opt.variational_coefs
	variational_dependent_h=opt.variational_dependent_h
	variational_dist=opt.variational_dist
	variational_random_padding=opt.variational_random_padding
	resolution_decoder=opt.resolution_decoder
	aleatoric_uncertainty=opt.aleatoric_uncertainty
	fourier_features=opt.fourier_features
	random_fourier=opt.random_fourier
	conditional_attention_style=opt.conditional_attention_style
	epochs=opt.epochs
	batch_size=opt.batch_size
	na_path_eeg=opt.na_path_eeg
	na_path_fmri=opt.na_path_fmri
	gpu_mem=opt.gpu_mem
	verbose=opt.verbose
	save_metrics=opt.save_metrics
	metrics_path=opt.metrics_path
	T=opt.T
	seed=opt.seed
	run_eagerly=opt.run_eagerly

	#assertion
	setting=dataset
	if(TRs>1):
		raise NotImplementedError
	if(topographical_attention):
		setting+="_topographical_attention"
	if(channel_organization):
		assert run_eagerly
		setting+="_channel_organization"
	if(random_fourier):
		assert fourier_features, "To run random_fourier, fourier_features need to be active"
		setting+="_random"
	if(fourier_features):
		setting+="_fourier_features"
	if(conditional_attention_style):
		assert topographical_attention, "To run conditional_attention_style, topographical_attention needs to be active"
		setting+="_attention_style"
	if(padded):
		assert not variational, "No variational model along with padded version of filling with zeros"
		assert type(resolution_decoder) is float, "There needs to be a specification of the lower resolution"
		setting+="_padded"
	if(variational):
		assert variational_coefs, "Need to be specified number of coefs, always upsampling for now, set issue to allow better implementation"
		setting+="_variational"
	if(type(variational_dist) is str):
		assert variational_dist in ["Normal", "VonMises"]
		if(variational):
			setting+="_"+variational_dist
	if(variational_dependent_h is None):
		variational_dependent_h=1
	if(variational_dependent_h > 1 and variational):
		setting+="_dependent_h_"+str(variational_dependent_h)
	if(type(variational_coefs) is str):
		assert variational, "Only done with variational flag set to True"
		variational_coefs=tuple(map(int ,variational_coefs.split(",")))
		assert len(variational_coefs) == 3, "Needs to specify all dimensions"
		assert type(variational_coefs[0]) is int and type(variational_coefs[1]) is int and type(variational_coefs[2]) is int, "Integers"
		assert variational_coefs[0] > 0 and variational_coefs[1] > 0 and variational_coefs[2] > 0, "Positive integers"
		setting+="_"+str(variational_coefs[0])+"x"+str(variational_coefs[1])+"x"+str(variational_coefs[2])
	if(type(resolution_decoder) is float):
		assert resolution_decoder > 1, "Resolution decoder needs to be \in [1,+\infty]"
		setting+="_res_"+"{:.1f}".format(resolution_decoder)
	if(variational_random_padding):
		assert variational, "Only done with variational flag set to True"
		setting+="_random_padding"



	return mode, dataset, TRs, topographical_attention, channel_organization, padded, variational, variational_coefs, variational_dependent_h, variational_dist, variational_random_padding, resolution_decoder, aleatoric_uncertainty, fourier_features, random_fourier, conditional_attention_style, epochs, batch_size, na_path_eeg, na_path_fmri, gpu_mem, verbose, save_metrics, metrics_path, T, seed, run_eagerly, setting

def clf_cv(opt):
	"""
	return the arguments in the opt variable,

	first it performs the according assertions needed for the script to execute

	"""

	dataset_synth=opt.dataset_synth
	dataset_clf=opt.dataset_clf
	feature_selection=opt.feature_selection
	segmentation_mask=opt.segmentation_mask
	style_prior=opt.style_prior
	padded=opt.padded
	variational=opt.variational
	variational_clf=opt.variational_clf
	variational_coefs=opt.variational_coefs
	variational_dependent_h=opt.variational_dependent_h
	variational_dist=opt.variational_dist
	variational_random_padding=opt.variational_random_padding
	resolution_decoder=opt.resolution_decoder
	aleatoric_uncertainty=opt.aleatoric_uncertainty
	view=opt.view
	folds=opt.folds
	fold=opt.fold
	n_processes=opt.n_processes
	epochs=opt.epochs
	gpu_mem=opt.gpu_mem
	path_save_network=opt.path_save_network
	run_eagerly=opt.run_eagerly
	seed=opt.seed
	path_labels=opt.path_labels
	save_explainability=opt.save_explainability

	setting=dataset_clf+"_synth_"+dataset_synth

	if(style_prior):
		setting+="_style_prior"
	if(variational_clf):
		setting+="_bayesian"
	if(padded):
		assert not variational, "No variational model along with padded version of filling with zeros"
		assert type(resolution_decoder) is float, "There needs to be a specification of the lower resolution"
		setting+="_padded"
	if(variational):
		assert variational_coefs, "Need to be specified number of coefs, always upsampling for now, set issue to allow better implementation"
		setting+="_variational"
	if(aleatoric_uncertainty):
		if(not variational):
			setting+="_variational_flipout"
	if(type(variational_dist) is str):
		assert variational_dist in ["Normal", "VonMises"]
		if(variational):
			setting+="_"+variational_dist
	if(variational_dependent_h is None):
		variational_dependent_h=1
	if(variational_dependent_h > 1 and variational):
		setting+="_dependent_h_"+str(variational_dependent_h)
	if(type(variational_coefs) is str):
		assert variational, "Only done with variational flag set to True"
		variational_coefs=tuple(map(int ,variational_coefs.split(",")))
		assert len(variational_coefs) == 3, "Needs to specify all dimensions"
		assert type(variational_coefs[0]) is int and type(variational_coefs[1]) is int and type(variational_coefs[2]) is int, "Integers"
		assert variational_coefs[0] > 0 and variational_coefs[1] > 0 and variational_coefs[2] > 0, "Positive integers"
		setting+="_"+str(variational_coefs[0])+"x"+str(variational_coefs[1])+"x"+str(variational_coefs[2])
	if(type(resolution_decoder) is float):
		assert resolution_decoder > 1, "Resolution decoder needs to be \in [1,+\infty]"
		setting+="_res_"+"{:.1f}".format(resolution_decoder)
	if(variational_random_padding):
		assert variational, "Only done with variational flag set to True"
		setting+="_random_padding"
	if(feature_selection):
		setting+="_feature_selection"
	if(segmentation_mask):
		setting+="_segmentation_mask"
	if(n_processes>1):
		assert gpu_mem/n_processes < 7000, "Should not be superior than memory of the GPU"

	if(view!="fmri"):
		setting=dataset_clf+"_"+view
		
	return setting,dataset_synth,dataset_clf,feature_selection,segmentation_mask,style_prior,padded,variational,variational_clf,variational_coefs,variational_dependent_h,variational_dist,variational_random_padding,resolution_decoder,aleatoric_uncertainty,view,fold,folds,n_processes,epochs,gpu_mem,path_save_network,seed,run_eagerly,path_labels,save_explainability