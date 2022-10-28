def save_state(tf, opt_instance, opt_config_path, gen_config_path):
	import pickle

	#optimizer_write(pickle, opt_config_path, opt_instance)

	generator_write(tf, pickle, gen_config_path)


def setup_state(tf, opt_instance, opt_config_path, gen_config_path):
	import pickle

	#optimizer = optimizer_read(tf, pickle, opt_config_path, opt_instance)

	generator_read(tf, pickle, gen_config_path)

	#return optimizer

def generator_write(tf, pickle, gen_config_path):

	gen = tf.random.get_global_generator()

	file_loader = open(gen_config_path,'wb')
	pickle.dump(gen, file_loader)
	file_loader.close()


def generator_read(tf, pickle, gen_config_path):
	#load it
	with open(gen_config_path, 'rb') as file_loader:
		gen = pickle.load(file_loader)

	tf.random.set_global_generator(gen)

def optimizer_write(pickle, file_path, opt):
	dict_class = opt.__dict__

	#remove clip lambda functions
	for key in list(dict_class.keys()):
		if("clip" in key):
			dict_class.pop(key, None)
			
	opt_file = open(file_path,'wb')
	pickle.dump(dict_class, opt_file)
	opt_file.close()

def optimizer_read(pickle, file_path, opt):
	dict_class = opt.__dict__

	#remove clip lambda functions
	for key in list(dict_class.keys()):
		if(not "clip" in key):
			dict_class.pop(key, None)

	#load it
	with open(file_path, 'rb') as file_loader:
		dict_class_opt = pickle.load(file_loader)

	opt.__dict__ = {**dict_class_opt, **dict_class}

	return opt	