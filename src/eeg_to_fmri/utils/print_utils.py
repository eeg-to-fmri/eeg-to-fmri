




def print_message(m, file_output=None, verbose=False, end="\n"):

	if(verbose):
		if(file_output==None):
			print(m, flush=True, end=end)
		else:
			print(m, file=file_output, flush=True, end=end)