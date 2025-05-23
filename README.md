| **Authors**  | **Project** | **License** |
|:------------:|:-----------:|:-----------:|
| [**N. Curti**](https://github.com/Nico-Curti) | **gbmseg**<br>[![American Journal of Nephrology](https://img.shields.io/badge/AJN-10.1159.000542658-g.svg)](https://karger.com/ajn/article-abstract/doi/10.1159/000542658/925719/Glomerular-basement-membrane-thickness-estimation?redirectedFrom=fulltext) | [![license](https://img.shields.io/github/license/Nico-Curti/glomerular-basement-membrane.svg)](https://github.com/Nico-Curti/glomerular-basement-membrane/blob/main/LICENSE) |

[![GitHub pull-requests](https://img.shields.io/github/issues-pr/Nico-Curti/glomerular-basement-membrane.svg?style=plastic)](https://github.com/Nico-Curti/glomerular-basement-membrane/pulls)
[![GitHub issues](https://img.shields.io/github/issues/Nico-Curti/glomerular-basement-membrane.svg?style=plastic)](https://github.com/Nico-Curti/glomerular-basement-membrane/issues)

[![GitHub stars](https://img.shields.io/github/stars/Nico-Curti/glomerular-basement-membrane.svg?label=Stars&style=social)](https://github.com/Nico-Curti/glomerular-basement-membrane/stargazers)
[![GitHub watchers](https://img.shields.io/github/watchers/Nico-Curti/glomerular-basement-membrane.svg?label=Watch&style=social)](https://github.com/Nico-Curti/glomerular-basement-membrane/watchers)

<a href="https://github.com/UniboDIFABiophysics">
  <div class="image">
    <img src="https://cdn.rawgit.com/physycom/templates/697b327d/logo_unibo.png" width="90" height="90">
  </div>
</a>

# GBM Image Analysis

## Glomerular Basement Membrane Segmentation and Thickness Estimation

Official implementation of the pipeline published on [Americal Journal of Nephrology](https://doi.org/10.1159/000542658) by Curti et al. [![American Journal of Nephrology](https://img.shields.io/badge/AJN-10.1159.000542658-g.svg)](https://karger.com/ajn/article-abstract/doi/10.1159/000542658/925719/Glomerular-basement-membrane-thickness-estimation?redirectedFrom=fulltext).

* [Overview](#overview)
* [Table of contents](#table-of-contents)
* [Contribution](#contribution)
* [Authors](#authors)
* [License](#license)
* [Acknowledgment](#acknowledgment)
* [Citation](#citation)

### :tada: Important updates 20/05/25 :tada:

With the new version of the `GBM` dataset, we improved the segmentation model changing its architecture to a more robust U-2-Net (ref. [here](https://github.com/xuebinqin/U-2-Net)) and increasing the amount of training samples to 326 images (with a total of more than 4k disjoint patches)!
The new version of the model is based on PyTorch and it provides the ONNX support for a faster computational time.
The example notebook for the inference of new segmentation are updated according to this new model.

## Overview

In this work we introduce a fully automated pipeline for the segmentation of GBM and successive thickness estimation, starting from TEM images.
This method is based on an active semi-supervised learning training procedure of a convolutional neural network model.
Starting from the areas automatically identified by the model, we provide a robust measurement of membrane thickness using pixels distance matrix and computer vision techniques.
Using these values, we trained a machine learning model to automatically determine the GBM thickness.
Starting from a small subset of images manually annotated by expert pathologists, we performed a series of active semi-supervised learning rounds of training to obtain the complete annotation of the dataset.
The obtained segmentations were used for the automated estimation of GBM thickness via computer vision algorithms and compared with manual measurements, obtaining a correlation of Pearson's R2 of 0.85.
The proposed pipeline obtained state-of-the-art performance in GBM segmentation, proving its robustness under image variations, such as magnification, contrast, and complex geometrical shapes.

## Table of contents

Description of the folders related to the project

| **Directory** |  **Description**                                                                |
|:--------------|:--------------------------------------------------------------------------------|
| [notebook](https://github.com/Nico-Curti/glomerular-basement-membrane/blob/main/notebook)       | `Jupyter` notebook with the list of examples about the segmentation model and thickness estimation. |
| [checkpoints](https://github.com/Nico-Curti/glomerular-basement-membrane/blob/main/checkpoints) | Segmentation model checkpoint weights obtained after 5 rounds of ASSL training |

## Contribution

Any contribution is more than welcome :heart:. Just fill an [issue](https://github.com/Nico-Curti/glomerular-basement-membrane/issues/new/choose) or a [pull request](https://github.com/Nico-Curti/glomerular-basement-membrane/compare) and we will check ASAP!

See [here](https://github.com/Nico-Curti/glomerular-basement-membrane/blob/main/.github/CONTRIBUTING.md) for further informations about how to contribute with this project.

## Authors

* <img src="https://avatars0.githubusercontent.com/u/24650975?s=400&v=4" width="25px"> [<img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="27px">](https://github.com/Nico-Curti) [<img src="https://cdn.rawgit.com/physycom/templates/697b327d/logo_unibo.png" width="25px">](https://www.unibo.it/sitoweb/nico.curti2) **Nico Curti**

See also the list of [contributors](https://github.com/Nico-Curti/glomerular-basement-membrane/contributors) [![GitHub contributors](https://img.shields.io/github/contributors/Nico-Curti/glomerular-basement-membrane.svg?style=plastic)](https://github.com/Nico-Curti/glomerular-basement-membrane/graphs/contributors/) who participated in this project.

## License

The `gbm-seg` package is licensed under the [GPL-3.0 License](https://github.com/Nico-Curti/glomerular-basement-membrane/blob/main/LICENSE)

### Acknowledgment

Thanks goes to all contributors of this project.

### Citation

If you have found `gbm-seg` helpful in your research, please consider citing the original paper about the wound image segmentation

```BibTex
@article{Curti2024,
  author={Curti, Nico and Carlini, Gianluca and Valente, Sabrina and Giampieri, Enrico and Merlotti, Alessandra and Dall'Olio, Daniele and Sala, Claudia and Marcelli, Emanuela and La Manna, Gaetano and Castellani, Gastone and Pasquinelli, Gianandrea},
  title={Glomerular basement membrane thickness estimation and stratification via active semi-supervised learning model},
  journal={American Journal of Nephrology},
  year={2025},
  month={April},
  day={16},
  abstract={Introduction: The measure of Glomerular Basement Membrane (GBM) thickness is used as diagnostic criteria for kidney glomerular diseases. The GBM thickness measurement, a time-consuming task, is performed by expert pathologists on transmission electron microscopy (TEM) images, therefore, it is affected by subjectivity and reproducibility issues. Methods: Here we introduce a fully automated pipeline for the GBM segmentation and successive thickness estimation, starting from TEM images. This method is based on an active semi-supervised learning training procedure of a convolutional neural network model. Starting from the areas automatically identified by the model, we provide a robust measurement of membrane thickness using pixels distance matrix and computer vision techniques. Using these values, we trained a machine learning model to automatically determine the GBM thickness. To verify the accuracy of the method, we compared the predicted results with the full iconographic materials and diagnostic record reports from 42 renal biopsies having normal-thick (n. 21), thin- (n. 10), thick-GBM (n. 11). Results: The obtained segmentations were used for the automated estimation of GBM thickness via computer vision algorithms and compared with manual measurements, obtaining a correlation of Pearson’s R2 of 0.85. The GBM thickness was stratified into 3 classes, namely normal, thin, thick with a 0.63 Matthews correlation coefficient and a 0.76 accuracy. Conclusion: The proposed pipeline obtained state-of-the-art performance in GBM segmentation, proving its robustness under image variations, such as magnification, contrast, and complex geometrical shapes. Automated measures could assist clinicians in standard clinical practice speeding up routine procedures with high diagnostic accuracy.},
  issn = {0250-8095},
  doi = {10.1159/000542658},
  url = {https://doi.org/10.1159/000542658},
  eprint = {https://karger.com/ajn/article-pdf/doi/10.1159/000542658/4365803/000542658.pdf},
}
```

or just this repository

```BibTex
@misc{gbm-seg,
  author = {Curti, Nico},
  title = {{Glomerular Basement Membrane Segmentation and Thickness Estimation}},
  year = {2024},
  url = {https://github.com/Nico-Curti/glomerular-basement-membrane},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Nico-Curti/glomerular-basement-membrane}}
}
```

