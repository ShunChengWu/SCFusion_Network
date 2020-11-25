![teaser](https://github.com/ShunChengWu/SCFusion_Network/blob/main/img/landscape_teaser.png)

This is the back-end network of SCFusion [repo](https://github.com/ShunChengWu/SCFusion) [[paper](https://arxiv.org/abs/2010.13662)].


We follow the data structure as the [edge-connect](https://github.com/knazeri/edge-connect) repository 

# Install
```
# cuda version: nvcc --version
conda create --name scfusion pytorch torchvision tensorboard pillow==6.1 cudatoolkit=YOUR_CUDA_TOOLKIT_VERSION  pyyaml -c pytorch -c conda-forge
conda activate scfusion
```
# Build Meshing tool
The meshing tool allows converting the input/output files into meshes

![example](https://github.com/ShunChengWu/SCFusion_Network/blob/main/img/example.png)

```
cd extension
python build.py install
```

* Train
```
python train.py --config /pth/to/config.yml
```
* test
```
python test.py --config /pth/to/config.yml
```
* sample  
need to install meshing tool extension
```
python sample.py --config /pth/to/config.yml
```
* trace  
export the trained model to be used on c++ 
```
python trace.py --config /pth/to/config.yml
```

The pre-trained model can be downloaded [here](https://github.com/pytorch/pytorch#from-source).   
