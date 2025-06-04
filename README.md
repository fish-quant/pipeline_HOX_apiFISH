# Pipeline_smFish_in apifih



## Getting started

  To download the pipeline_smFISH : git clone   ##########


## Install environments

1) Base environment: 

a) conda create --name base_env_apifish python=3.10.16
b) conda activate base_env_apifish
c) pip install apifish
c) python -m pip install cellpose[gui]
d) Other libraries to install in pipeline_fish (using pip install x)   where x = ipython, napari, nd2reader, ipykernel, readlif, ipywidgets
e) add the kernel to jupyter :
python -m ipykernel install --user --name base_env --display-name "base_env"

f) create other environments (for instance ufish_env): 
    conda activate base
    conda env create -f ufish_env.yml
    
    Optional: install ipykernel (conda install ipykernel) , add new kernel to your conda environment:   python -m ipykernel install --user --name environment_x --display-name "environment_x"
    Then you can test your api directly in the native environment environment_x, and then write a wrapper to do cross-environment function call.

g) test if cude is available, otherwise install it: python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
 
## Authors
Jacques Bourg @ Muller lab, institut pasteur. 04/06/25
 
 
