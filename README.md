# convert_tensor_record
This code converts the images to tfrecord file.
referance site is https://github.com/tensorflow/models/tree/master/research/slim

# Step 1. Settings: OS-Ubuntu 16.04
conda create -n tf-slim python=2.7 cudatoolkit=9.0 cudnn=7.3 tensorflow-gpu=1.10

conda list
->

_libgcc_mutex             0.1                        main  
_tflow_select             2.1.0                       gpu  
absl-py                   0.7.1                    py27_0  
astor                     0.7.1                    py27_0  
backports                 1.0                        py_2  
backports.weakref         1.0.post1                  py_1  
blas                      1.0                         mkl  
bzip2                     1.0.8                h7b6447c_0  
c-ares                    1.15.0               h7b6447c_1  
ca-certificates           2019.5.15                     0  
cairo                     1.14.12              h8948797_3  
certifi                   2019.6.16                py27_0  
cudatoolkit               9.0                  h13b8566_0  
cudnn                     7.3.1                 cuda9.0_0  
cupti                     9.0.176                       0  
enum34                    1.1.6                    py27_1  
ffmpeg                    4.0                  hcdf2ecd_0  
fontconfig                2.13.0               h9420a91_0  
freeglut                  3.0.0                hf484d3e_5  
freetype                  2.9.1                h8a8886c_1  
funcsigs                  1.0.2                    py27_0  
futures                   3.3.0                    py27_0  
gast                      0.2.2                    py27_0  
glib                      2.56.2               hd408876_0  
graphite2                 1.3.13               h23475e2_0  
grpcio                    1.16.1           py27hf8bcb03_1  
harfbuzz                  1.8.8                hffaf4a1_0  
hdf5                      1.10.2               hba1933b_1  
icu                       58.2                 h9c2bf20_1  
intel-openmp              2019.4                      243  
jasper                    2.0.14               h07fcdf6_1  
jpeg                      9b                   h024ee3a_2  
libedit                   3.1.20181209         hc058e9b_0  
libffi                    3.2.1                hd88cf55_4  
libgcc-ng                 9.1.0                hdf63c60_0  
libgfortran-ng            7.3.0                hdf63c60_0  
libglu                    9.0.0                hf484d3e_1  
libopencv                 3.4.2                hb342d67_1  
libopus                   1.3                  h7b6447c_0  
libpng                    1.6.37               hbc83047_0  
libprotobuf               3.8.0                hd408876_0  
libstdcxx-ng              9.1.0                hdf63c60_0  
libtiff                   4.0.10               h2733197_2  
libuuid                   1.0.3                h1bed415_2  
libvpx                    1.7.0                h439df22_0  
libxcb                    1.13                 h1bed415_1  
libxml2                   2.9.9                hea5a465_1  
markdown                  3.1.1                    py27_0  
mkl                       2019.4                      243  
mkl_fft                   1.0.12           py27ha843d7b_0  
mkl_random                1.0.2            py27hd81dba3_0  
mock                      3.0.5                    py27_0  
ncurses                   6.1                  he6710b0_1  
numpy                     1.16.4           py27h7e9f1db_0  
numpy-base                1.16.4           py27hde5b4d6_0  
opencv                    3.4.2            py27h6fd60c2_1  
openssl                   1.1.1c               h7b6447c_1  
pcre                      8.43                 he6710b0_0  
pip                       19.1.1                   py27_0  
pixman                    0.38.0               h7b6447c_0  
protobuf                  3.8.0            py27he6710b0_0  
py-opencv                 3.4.2            py27hb342d67_1  
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
xz                        5.2.4                h14c3975_4  
zlib                      1.2.11               h7b6447c_3  
zstd                      1.3.7                h0b5b093_0



# Step 2. Use
conda activate tf-slim

cd convert_tensor_record

python DataSetting.py --dataset_name=RAF --dataset_dir=your_save_directory_path --label_file=your_label_file_path --split_name=test

detail description -> see DataSetting.py

label file sample:

/media/kmc/2TB_HD/dataset/RAF/experiment/aligned/test_5classes/neutral/basic_test_2642_E0.jpg 0
/media/kmc/2TB_HD/dataset/RAF/experiment/aligned/test_5classes/neutral/basic_test_2390_E0.jpg 0
/media/kmc/2TB_HD/dataset/RAF/experiment/aligned/test_5classes/neutral/basic_test_2391_E0.jpg 0



