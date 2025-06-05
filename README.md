# Pipeline_HOX_apiFISH



## Getting started

To download the pipeline_smFISH : git clone https://github.com/fish-quant/pipeline_HOX_apiFISH.git


## Install environments

        1) Base environment: 

        a) conda create --name base_env_apifish python=3.10.16

        b) conda activate base_env_apifish

        c) pip install apifish

        d) python -m pip install cellpose[gui]==3.1.0

        e) Other libraries to install in pipeline_fish (using pip install x)
        where x = ipython, napari, nd2reader, ipykernel, readlif, ipywidgets

        f) add the kernel to jupyter :
        python -m ipykernel install --user --name base_env_apifish --display-name "base_env_apifish"

        2) Create second environment (ufish_env): 
            conda activate base
            conda env create -f ufish_env.yml
            
            Optional: install ipykernel (conda install ipykernel), 
            add new kernel to your conda environment:   
            python -m ipykernel install --user --name environment_x --display-name "environment_x"
            
            Then you can test your api directly in the native environment environment_x, 
            and then write a wrapper to do cross-environment function call.

        3) Test if cuda is available otherwise install it.
        
        python -c "import torch;
        print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
 
## Authors
Jacques Bourg @ Florian Muller lab. Institut Pasteur. 04/06/25
 
 
