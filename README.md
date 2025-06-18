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
            
            e) add new kernel to your conda environment:   
            python -m ipykernel install --user --name ufish_env --display-name "ufish_env"
            

        3) Test if cuda is available otherwise install it.
        
        python -c "import torch;
        print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
 
 
## Code execution

        1) In Linux/Mac, open a terminal. In Windows open the Anaconda Prompt. Place the terminal's current working directory 
        at the pipelines root "../pipeline_HOX_apiFISH".
        
        2) Place yourself in the conda environment:
        
        conda activate base_env_apifish
        
        3) Launch the jupyter server:
        
        jupyter notebook.
        
        4 ) Execute the pipeline in the order given by the diagram.
        
        5)  All jupyter notebooks should be run in the environment "base_env_apifish", except the notebook called Spot_detection_part1. 
        
 
## Authors
Jacques Bourg @ Florian Muller lab. Institut Pasteur. 04/06/25
 
 
