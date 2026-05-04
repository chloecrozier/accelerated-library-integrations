nvidia smi screenshot

gh auth login # used ssh auth

git clone the repo
cded into RAPIDS-cuDF

ccrozier@spark-1b89:~$ conda create -n rapids
conda init
ccrozier@spark-1b89:~$ conda activate rapids
(rapids) ccrozier@spark-1b89:~$ 


conda install -c rapidsai libcudf
conda install -c rapidsai pylibcudf
conda install -c rapidsai cudf
conda install -c rapidsai cudf-polars
conda install -c rapidsai dask-cudf


cudf installed screenshot

python install_verification.py 

screenshot for install verfifcaton resutls


conda install -c conda-forge jupyterlab

jupyter lab --no-browser --ip=127.0.0.1 --port=8888 basic_uses.ipynb
ssh -N -L 8888:localhost:8888 <username>@<remote-host>

then go to http://localhost:8888/


jupyter lab screenshot

