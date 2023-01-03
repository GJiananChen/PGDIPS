## PGDIPS: Physics-guided deep image prior for stain deconvolution

Pytorch implementation for the manuscript "[General stain deconvolution of histopathology images with physics-guided deep learning](https://www.biorxiv.org/content/10.1101/2022.12.06.519385v1)" 


### Installation
Python version requirement: 3.9+

The easiest installation is through Anaconda or [anaconda/miniconda](https://docs.conda.io/en/latest/miniconda.html)

```
  $ conda env create --name pgdips 
  $ conda activate pgdips
  $ pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
  $ pip3 install -r requirements.txt
```

> **_NOTE:_** Please check if your graphics card is compatible with the following the packages: `cudatoolkit=11.6` and `pytorch 1.13.0`. If your GPU will not support them, you can downgrade `pytorch`, `cuda` and `cudnn`.

Alternatively, you can install all required packages specified in `environment.yml`.
```
  $ conda env create --name pgdips --file environment.yml
  $ conda activate pgdips
```

## Training on provided images

Train PGDIPS on provided target images:

```
  $ python pgdips.py --config=he.cfg
```

The following example target images are available:

```
[--config]      # he.cfg, ihc.cfg, alcianblue.cfg, DeepLIIF6.cfg, fibronectin.cfg, SMACD34.cfg, trichrome.cfg
```

## Training on your own data (minimum 10x resolution, ideally 20x-40x) by creating a configuration file 

### 1. Place patch files as `input\[image_name]`. Supported formats are `.jpg`, `.png`, `.tiff` and `.tif`.

### 2. Create a configuration file for your own data based on the following template and save it as `config\[config_name]`.

```
[system]
input = [image_name]
input_folder = ./input
output_folder = ./output

[preprocess]
input_depth = 3
input_size = 512
roi_selection = centercrop
x_coor_start = 1
y_coor_start = 1

[stain]
num_stains = 2
fixed_color = False
default_stain_vector_1 = 0.60, 0.75, 0.29
default_stain_vector_2 = 0.21, 0.91, 0.36
use_spectral_correction_factor = False

[network]
iterations = 6000
keep_color_iterations = 2000
init_learning_rate = 0.0005
plot_during_training = True
show_every = 400

[loss]
weight_l1 = 1
weight_exclusion = 0.3
weight_mse = 1
```

#### Hyperparameter Legend

- `input = [image_name]` # name of the input/target image in the `input_folder`
- `input_folder = ./input` # where input images are stored
- `output_folder = ./output` # where output files should be stored
- `input_depth = 3` # input image color channels, `1` for grayscale and `3` for rgb
- `input_size = 512` # `256`, `512`, `1024`, etc # dimensions of input image
- `roi_selection = centercrop` # how to select from the target image, pick from the following: `centercrop`, `crop`, `resize`
- `x_coor_start = 1` # x-coordinate of the top left pixel for cropping, only applicable when `roi_selection` is `crop`
- `y_coor_start = 1` # y-coordinate of the top left pixel for cropping, only applicable when `roi_selection` is `crop`
- `num_stains = 2` # number of stains, `2` or `3` 
- `fixed_color = False` # whether to NOT learn stain color vectors. Only use `True` if you know that the default stain vectors are the true stain color vectors
- `default_stain_vector_1 = 0.60, 0.75, 0.29` # default stain color vector for stain 1, no need to change these
- `default_stain_vector_2 = 0.21, 0.91, 0.36` # default stain color vector for stain 2 no need to change these
- `use_spectral_correction_factor = False` # `True` or `False`, whether to apply the spectral correction factor
- `iterations = 6000` # number of iterations, `6000` iters is a good place to start
- `keep_color_iterations = 2000` # how long are the stain vectors regulated by the default stain vectors
- `plot_during_training = True` # save intermediate plots?
- `show_every = 400` # how often to save intermediate plots
- `weight_l1 = 1` # weight for the l1 reconstruction losss
- `weight_exclusion = 0.3` # weight for exlusion loss, increase this if concentration maps overlap and decrease this when one or more c-maps are empty
- `weight_mse = 1` # weight for the l2 color fixing loss

> **_NOTE:_** The primary criteria for selecting ROI is to make sure that the ROI contains enough (and ideally relatively equal amounts of) concentrations of all the target stains. `centercrop` is the default option. `resize` is not recommended as it changes the spectral bias (_i.e._ deep image prior) of the image and may require chaning the network structure.

> **_NOTE:_** Input size determines runtime and VRAM bottleneck. A GPU with less than 6 GB VRAM can handle input size of 1024 but 512 is recommended.

### 3. Train a PGDIPS model.

```
$ python pgdips.py --config=[config_name]
```

The cropped original image will be automatically opened once you launch the command. The default output files (saved in the output folder) include the following:
1. `CONFIGNAME_Con1_Iterations.png`: concentration map for estimated stain 1
2. `CONFIGNAME_Con2_Iterations.png`: concentration map for estimated stain 2
3. `CONFIGNAME_Stained1_Iterations.png`: colored concentration map (concentration map multiplied by the corresponding color vector) for estimated stain 1
4. `CONFIGNAME_Stained2_Iterations.png`: colored concentration map for estimated stain 2
5. `CONFIGNAME_All_Iterations.png`: reconstructed image with background illunimation removed
6. `CONFIGNAME_WB_Iterations.png`: reconstructed image (including estimated background illunimation)
7. `CONFIGNAME_original.png`: cropped/resized original image
8. `CONFIG_mse.png`: a graph for the mean squared error between original image and reconstructed image through different iterations. 
9. `CONFIGNAME_out.csv`: a spreadsheet containing estimated physics parameters at different iterations, including S1R, S1G, S1B (color vectors for stain 1 in RGB channels), S2R, S2G, S2B (color vectors for stain 2)，SCF1, SCF2 (spectral correction factor for stain 1 and 2, respectively), BGR, BGG, BGB (Background illunimation 
in RGB channels)

## What can I do if I am not happy with PGDIPS deconvolution results?
1. **Double check the quality of input image**: Please double check the preprocessed (cropped) image for stain deconvolution, which is stored in the output folder. PGDIPS only has issues in two known cases: 1) when the colors of two stains are too similiar 2) when the distribution of two stains are too imbalanced (for example when there are too much hematoxylin and too little DAB stain in the input image).

2. **Run PGDIPS again with parameters learned from the first run**: After running PGDIPS the physics parameters such as stain color vectors and background correction factors are saved in the output folder. Running PGDIPS again using these parameters (for example by setting default color vectors 1 and 2 to the estimated color vectors from pervious rounds) will help improve the assignment of structures to correct colors, as the model will start from correct colors at the beginning of training.

3. **Tune the weights of the different loss functions**: We found that `weight_l1=1` and `weight_mse=1` works in all our experiments. `weight_exclusion` can have a big impact on the estimated concentration maps. If one of the concentration maps is too blurry or empty, try tuning `weight_exclusion` down. 

4. **Rerun PGDIPS**: As the input of DIP modules are random noise, when the initialization of the random noise went wrong, the concentration maps may contains black hole artifacts. This is very rare and can usually solved by running PGDIPS again.


## License

© [Sunnybrook Research Institute ](https://sunnybrook.ca/research/) - PGDIPS code is distributed under [**BSD 3 with Commons Clause**  license](https://github.com/GJiananChen/PGDIPS/blob/master/LICENSE.md), and is available for non-commercial academic purposes.

## Contact

Bugs can be reported in the [GitHub Issues](https://github.com/GJiananChen/PGDIPS/issues) tab.

If you have any questions about this code, I am happy to answer your issues or emails (to [chenjn2010@gmail.com](mailto:chenjn2010@gmail.com)).

## Acknowledgements

The work conducted by [Jianan Chen](https://gjiananchen.github.io/) was funded by grants from the [Martel lab](https://github.com/martellab-sri).

This code is inspired by [DoubleDIP](https://github.com/yossigandelsman/DoubleDIP).

## Citation

If you use the code or results in your research, please use the following BibTeX entry.

```
@article {ChenPGDIPS,
	author = {Chen, Jianan and Liu, Lydia Y and Han, Wenchao and Wang, Dan and Cheung, Alison M and Tsui, Hubert and Martel, Anne L},
	title = {General stain deconvolution of histopathology images with physics-guided deep learning},
	elocation-id = {2022.12.06.519385},
	year = {2022},
	doi = {10.1101/2022.12.06.519385},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2022/12/10/2022.12.06.519385},
	eprint = {https://www.biorxiv.org/content/early/2022/12/10/2022.12.06.519385.full.pdf},
	journal = {bioRxiv}
}

```

## 
