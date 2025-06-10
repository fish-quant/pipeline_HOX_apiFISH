# Pipeline_HOX_apiFISH



## Getting started

        To download the pipeline_smFISH : git clone https://github.com/fish-quant/pipeline_HOX_apiFISH.git


## Install environments

        1) Base environment: 

        a) conda create --name base_env_apifish python=3.11

        b) conda activate base_env_apifish

        c) pip install -r requirements_base_env_apifish.txt    
        
        d) add the kernel to jupyter :
        python -m ipykernel install --user --name base_env_apifish --display-name "base_env_apifish"

        2) Create second environment (ufish_env): 
        
            a) conda activate base
            
            b) conda create --name ufish_env python=3.11
            
            c) conda install pip
            
            d) pip install -r requirements_ufish.txt
            
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
 
 
