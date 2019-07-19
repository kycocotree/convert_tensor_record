# convert_tensor_record
This code converts the images to tfrecord file.
referance site is https://github.com/tensorflow/models/tree/master/research/slim

# Step 1. Settings: OS-Ubuntu 16.04

The table below is anaconda envs package list to implement this code.

Name                    Version                   Build  Channel

_libgcc_mutex             0.1                        main  
_tflow_select             2.1.0                       gpu  
absl-py                   0.7.1                    py27_0  
astor                     0.7.1                    py27_0  
backports                 1.0                        py_2  
backports.weakref         1.0.post1                  py_1  
blas                      1.0                         mkl  
c-ares                    1.15.0               h7b6447c_1  
ca-certificates           2019.5.15                     0  
certifi                   2019.6.16                py27_0  
cudatoolkit               9.0                  h13b8566_0  
cudnn                     7.3.1                 cuda9.0_0  
cupti                     9.0.176                       0  
enum34                    1.1.6                    py27_1  
funcsigs                  1.0.2                    py27_0  
futures                   3.3.0                    py27_0  
gast                      0.2.2                    py27_0  
grpcio                    1.16.1           py27hf8bcb03_1  
intel-openmp              2019.4                      243  
libedit                   3.1.20181209         hc058e9b_0  
libffi                    3.2.1                hd88cf55_4  
libgcc-ng                 9.1.0                hdf63c60_0  
libgfortran-ng            7.3.0                hdf63c60_0  
libprotobuf               3.8.0                hd408876_0  
libstdcxx-ng              9.1.0                hdf63c60_0  
markdown                  3.1.1                    py27_0  
mkl                       2019.4                      243  
mkl_fft                   1.0.12           py27ha843d7b_0  
mkl_random                1.0.2            py27hd81dba3_0  
mock                      3.0.5                    py27_0  
ncurses                   6.1                  he6710b0_1  
numpy                     1.16.4           py27h7e9f1db_0  
numpy-base                1.16.4           py27hde5b4d6_0  
openssl                   1.1.1c               h7b6447c_1  
pip                       19.1.1                   py27_0  
protobuf                  3.8.0            py27he6710b0_0  
python                    2.7.16               h9bab390_0  
readline                  7.0                  h7b6447c_5  
setuptools                41.0.1                   py27_0  
six                       1.12.0                   py27_0  
sqlite                    3.29.0               h7b6447c_0  
tensorboard               1.10.0           py27hf484d3e_0  
tensorflow                1.10.0          gpu_py27h6f941b3_0  
tensorflow-base           1.10.0          gpu_py27h6ecc378_0  
tensorflow-gpu            1.10.0               hf154084_0  
termcolor                 1.1.0                    py27_1  
tk                        8.6.8                hbc83047_0  
werkzeug                  0.15.4                     py_0  
wheel                     0.33.4                   py27_0  
zlib                      1.2.11               h7b6447c_3

# Step 2. Use
python DataSetting.py --dataset_name=RAF --dataset_dir=your_save_directory_path --num_shard=5 --image_width=227 --image_height=227 --ratio_val=0.1 --label_file=your_label_file_path

flag description:
dataset_name: A name of the dataset.
dataset_dir: The save path of the tensor record file.
num_shards: A number of sharding for TFRecord files(integer).
image_width: A number of width to resize(integer). #network input size.
image_height: A number of height to resize(integer). #network input size.
ratio_val: A ratio of validation datasets for TFRecord files(flaot, 0 ~ 1).
label_file: The path of the label file.

label file sample:

/media/kmc/2TB_HD/dataset/RAF/experiment/aligned/test_5classes/neutral/basic_test_2642_E0.jpg 0
/media/kmc/2TB_HD/dataset/RAF/experiment/aligned/test_5classes/neutral/basic_test_2390_E0.jpg 0
/media/kmc/2TB_HD/dataset/RAF/experiment/aligned/test_5classes/neutral/basic_test_2391_E0.jpg 0


detail description -> see DataSetting.py
