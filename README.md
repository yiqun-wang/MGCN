# MGCN

<p align="center">
  <img src="outputs/demo.gif" alt="demo">
</p>

This code implements Multiscale Graph Convolutional Network (MGCN) in our SIGGRAPH 2020 paper:

["MGCN: Descriptor Learning using Multiscale GCNs"](https://arxiv.org/abs/2001.10472) 

by Yiqun Wang, Jing Ren, Dong-Ming Yan, Jianwei Guo, Xiaopeng Zhang, Peter Wonka.

Please consider citing the above paper if this code/program (or part of it) benefits your project. 


## Environment
```bash	
	conda create -n MGCN python=3.7     # (options: 3.X)
	source activate MGCN                # (create and activate new environment if you use Anaconda)
	
	conda install pytorch=1.5.0 torchvision cudatoolkit=10.1 -c pytorch     # (options: 10.X)
	pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
	pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
	pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
	pip install torch-geometric
```

## Usage

1. Put models in the folder MGCN/datasets/Faust/data_mesh/ for FAUST dataset or MGCN/datasets/Scape/data_mesh/ for SCAPE dataset

2. generate Wavelet Energy Decomposition Signature (WEDS) descriptors and wavelets using Repository [WEDS](https://github.com/yiqun-wang/WEDS) or directly download processed data (traning.pt and test.pt) below and put them into the folder MGCN/datasets/Faust/processed/ or MGCN/datasets/Scape/processed/.

3. Training examples
  
```bash
	# for FAUST
	python MGCN_FAUST.py
	# for SCAPE
	python MGCN_SCAPE.py
```

4. Restore checkpoint and generate descriptors
  
```bash
	# for FAUST
	python MGCN_FAUST.py --uc --gd -l --ln=mgcn_faust-300
	# for SCAPE
	python MGCN_SCAPE.py --uc --gd -l --ln=mgcn_scape-300
```

5. I recommend this [Repository](https://github.com/llorz/MatlabRenderToolbox) to visualize the descriptors.

## Processed data

| Processed Dataset | Download Link | Description |
|:-|:-|:-|
| FAUST original | [Google Drive](https://drive.google.com/open?id=1DWIvdqPDPNaf6ZYqMeeRZjASfdXU7Jz2), 1G | 75 models for training and 15 models for testing (6890 points) |
| FAUST 5 resolutions | [Google Drive](https://drive.google.com/open?id=1uzbsXSexjMoX9gTrzK7NqHhcljQfXhVj), 1G | 15 X 5 models for testing (6890, 8K, 10K, 12K, 15K points) |
| SCAPE remeshed | [Google Drive](https://drive.google.com/open?id=1d0MOVVcBt5y2dhPgIajmFAIuDH9ZodYo), 623M | 61 models for training and 10 models for testing (~5K points) |
| SCAPE original | [Google Drive](https://drive.google.com/open?id=1_Pu_zwabWpeB_7xh2IPkWOh6gBOZeuxi), 218M | 10 models for testing (12.5K points) |

	
## Cite

    @article{wang2020mgcn,
      title=      {MGCN: Descriptor Learning using Multiscale GCNs},
      author=     {Wang, Yiqun and Ren, Jing and Yan, Dong-Ming and Guo, Jianwei and Zhang, Xiaopeng and Wonka, Peter},
      journal=    {ACM  Trans. on Graphics (Proc. {SIGGRAPH})},
      year=       {2020},
    }

## License

This program is free software; you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation; either version 2 of 
the License, or (at your option) any later version. 
