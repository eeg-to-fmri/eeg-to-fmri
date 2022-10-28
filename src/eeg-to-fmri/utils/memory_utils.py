import resource

def limit_CPU_memory(bytes, n_processes):
	return
	soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
	resource.setrlimit(resource.RLIMIT_DATA, (bytes, hard))#heap of the process
	resource.setrlimit(resource.RLIMIT_AS, (1024*1024*1024*26, hard))#virtual memory allocation
	resource.setrlimit(resource.RLIMIT_RSS, (1024*1024*1024*8, 1024*1024*1024*16))#resident memory
	soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
	resource.setrlimit(resource.RLIMIT_NPROC, (n_processes,n_processes))
	#resource.setrlimit(resource.RLIMIT_SWAP, (1024*1024*1024*2, hard))