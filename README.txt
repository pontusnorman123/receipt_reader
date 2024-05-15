Step 1:

python3 -m venv env

step 2:

source env/bin/activate

step 3:

pip install -r requirements.txt

step 4: 

start server with: "gunicorn -w 4 -b 0.0.0.0:5000 receipt_reader_server:app"

step 5: run "python3 inference_with_server.py"

Right now receipt images are read from the "image_folder" and results are saved in a json file in the "results" folder. Note that all the current images in the "image_folder" are processed and are currently not deleted after being processed. 


Needs CUDA to run on GPU which could improve speed: https://pytorch.org/get-started/locally/
https://docs.nvidia.com/cuda/wsl-user-guide/index.html



