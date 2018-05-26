
RUNNING An IPYTHON NOTEBOOK ON THE SERVER WITH LOCAL BROWSER ACCESS:
	From Local:
		To open:
		ssh -N -f -L localhost:8888:localhost:8890 fpk20@mario.eng.cam.ac.uk (THIS HAS TO BE THE SAME AS ON THE server side)
		Now open your browser on the local machine and type in the address bar:
		localhost:8888
		To close:
		to display processess
			ps aux | grep localhost:8890
		kill process id
		kill -15 process_id
		REMEMBER TO kill the process
	From the server:
		check if ssh config is appropiate ( if you want yoshi you need to change it there)
		ssh fpk20@mario.eng.cam.ac.uk
		cd /scratch/fpk20
		ipython "notebook" --no-browser --port=8889


To copy a file from local to server side
	scp somefile fpk20@mario.eng.cam.ac.uk:/scratch/fpk20

TO CHECK THE PROCCESSES on server
	nvidia-smi

To create a virtualen GPU to install shit
	virtualenv gpu
	source gpu/bin/activate
	pip install tensorflow-gpu==1.4

To connect to ssh yoshi
	check if config 

OLD .bash_profile	
export PATH=$HOME/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/cuda-9.0/lib64/

export PATH=/usr/local/cuda/bin:$PATH
exportLIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zelda/fpk20/cuda-9.0/lib64
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda
