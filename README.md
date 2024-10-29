# Interpretable-by-design deep survival analysis for disease progression modeling

### Accepted at MICCAI 2024

This repository contains the code for the paper "Interpretable-by-design Deep Survival Analysis for Disease Progression Modeling". The code is based on PyTorch and is used to train and evaluate a deep learning model for survival analysis on fundus images. The model offers built-in interpretability by yielding an evidence map of local disease risk which is subsequently simply aggregated to a final risk prediction. The model is based on a combination of a Sparse BagNet and a Cox proportional hazards model. As a sample use case, it is trained to predict the risk of conversion to age-related macular degeneration at different time points and its performance is compared to that of black-box SOTA baseline models.

<br>

<img src="figures/Fig1.png" alt="Figure 1: Model architecture" width="700"/><br>
Figure 1: Model architecture. <br><br>

<img src="figures/Tab1.png" alt="Table 1: Model performance" width="500"/>

## Pre-requisites

- Obtain the AREDS data (see [Data](#data))
- Install the conda environment using `conda env create -f requirements.yml` (or use `requirements_without_R.yml` which installs much faster but lacks AUPRC metric support) and activate it with `conda activate amd-surv`.
- Log in to [wandb](https://wandb.ai) with the command `wandb login`. Create an account if you don't have one.

## Load and evaluate the pre-trained interpretable-by-design survival model on AREDS data

- Retrieve the model weights `7ufjvvnz_best.pth` from [here](https://zenodo.org/records/10787031?token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTcwOTczNTcxMSwiZXhwIjoxNzI4OTUwMzk5fQ.eyJpZCI6ImY4MzNhYWMxLTA1MDMtNGY4YS04ZDI0LWRlNGZjODIzYTdhMyIsImRhdGEiOnt9LCJyYW5kb20iOiI1ZTRjNTU3MWRiMTM1MzY2YTgyNDYwOTliYjYxOTQ1MSJ9.Qzh_-DCP6G_K2FgWz2lzn26GR3ojEf4r3k16lJmP73ZQNMJR_cvV8h321ggIQ46OtrMzQaOqLNclRfrEniSm7w) and place it in the `checkpoints` directory.
- Follow the instructions in `evaluate_survival.ipynb` to load the model and evaluate it.
- To evaluate the pre-trained baseline models, retrieve the other weights files and repeat the procedure.

## Alternatively, train the model yourself

The model can be trained and evaluated using the following command: <br>
`python train_and_evaluate_survival.py --config configs/sparsebagnet_cox.yml` <br>

Similarly, the baseline models can be trained and evaluated using the files in `configs/babenko/` and `configs/yan/`. The model configs serve to train one model for each inquired point in time (years 1 to 5).

## Modify the configuration

The training is configured using a yaml file. The following parameters can be set:
<details>

<summary>YAML</summary>

```yaml
# Path to the metadata csv file relative to the project directory.
metadata_csv: data/metadata_surv.csv    # input file with image paths etc. Make sure to select one with stereo images if selecting use_stereo_images: true below.

# Path to image files, relative to the data directory in dirs.yml.
image_dir: images-f2-1024px

### CNN training ###
cnn:
  project: wandb-project-name           # The name of the wandb project. This code requires wandb to be installed and logged in.
  run_id: none                          # Needed to load a checkpoint, optional
  resume_training: false                # Load model from last checkpoint, optional
  load_best_model: false                # Load model from best checkpoint, optional
  test_run: 
    enabled: false                      # If true, only use a small subset of the data   
    size: 100                           # Ignored if enabled=false
  train_set_fraction: 1.0               # For debuggung use <1.0. Overwrites test_run.size
  val_set_fraction: 1.0                 # For debugging use <1.0
  survival_times: [2, 4, 6, 8, 10]      # Times to evaluate the model at, in half-years. You can pass e.g. "[2]" in combination with "loss: clf" to train a classifier model for time 2.
  gpu: 0                                # Index of GPU to use
  seed: 123                             # Random seed for reproducibility
  batch_size: 8                         # Batch size
  num_epochs: 50                        # Max. number of epochs
  stop_after_epochs: 10                 # Stop training after this number of epochs without improvement
  num_workers: 8                        # Number of workers for data loading
  img_size: 350                         # Height and width of input images
  network: sparsebagnet-surv            # Combination of {resnet, sparsebagnet (or, equally: bagnet), inceptionv3} and {surv}
  lr: 0.000016                          # Learning rate
  sparsity_lambda: 0.000006             # Sparsity loss coefficient (only applied to BagNet models)
  loss: cox                             # cox or clf. Determines model type. clf is for classification models using ce-loss, trained for only one survival_time
  use_stereo_pairs: false               # If true, use stereo image pairs and average their predictions after non-linear activation. Use correct metadata_csv that includes images of both sides of an eye!
  model_selection_metric: ibs           # ibs, aunbc, auc or bce_loss
  eval_sets: [val, test]                # (list of) set(s) to evaluate the model on, e.g. "[val]"
  optimizer: adam                       # adam, adamw or sgd
  weight_decay: 0.0                     # Set to >0.0 to use weight decay
  scheduler: none                       # onecyclelr, cosineannealing, cosineannealingrestarts or none
  scheduler_cosine_len: 0               # Length of a cosine period in epoch units. If not passed, uses epochs/2. Ignored if scheduler is not cosineannealing or cosineannealingrestarts.
  warmup_epochs: 0                      # Number of epochs to warmup the learning rate, set to 0 for no warmup
  augmentation: true                    # If true, use data augmentation as in Huang et al. 2023
  balancing: false                      # Class balancing, currently not supported. Set to false
  num_classes: 12                       # AMD severity scale length for dataloader, do not change
```

</details>

<br>
The model takes below 12h to train and evaluate on a single NVIDIA GeForce RTX 2080 Ti GPU.

<h2 id="data">Data</h2>

Upon request, the AREDS data can be accessed from the [dbGaP repository](https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs000001.v3.p1). The tabular data needs to be parsed to one metadata file with the rows representing individual macula-centered (F2) images. <!--Throughout the study, two identifiers for participants have been used. Match participants to ID2 identifiers using the mapping file that comes with the AREDS data. <br>-->

The AREDS data directory as defined in `configs/dirs.yml` should contain:<br>
- a folder as set the model config (`image_dir`), the directory of image data. The images should be organized such that the metadata CSV image paths map to the image files, relative to the `image_dir`. <br>

The `<project>/data` directory should contain the following files:<br>
- `metadata_surv.csv`: Survival metadata table that maps screening data (event, duration) to image records. An example entry is provided in `data/metadata_surv example.csv` with important columns as follows:
  <details>

  <summary>Columns</summary>

  - patient_id, the eye identifier
  - visit_number, VISNO
  - image_eye, right or left eye
  - image_field, F2
  - image_side, RS or LS
  - image_file, file name
  - image_path, relative path incl. file name from data dir specified in `dirs.yml`
  - duration, relative time to the first visit where late AMD was diagnosed in units of visits (half years)
  - event, 1 if any record of this eye converted to late AMD, 0 else
  - diagnosis_amd_grade_12c, AMDSEVRE and AMDSEVLE: the AMD severity scale score starting at 0

  </details>

<ul><li><code>metadata_surv_stereo.csv</code>: Image metadata table with the same structure, but of records where both views (LS and RS) exist (for Babenko et al. baseline model).</li></ul><br>

Both datasets can be created from a parsed AREDS metadata file using `save_survival_metadata.ipynb`.

## Illustrative figures

<img src="figures/Fig2.png" alt="Figure 2: Example risk evidence maps and survival curve predictions" width="700"/><br>
Figure 2: Example risk evidence maps (a) and the resulting survival curve predictions (b). <br><br>

<img src="figures/Fig3.png" alt="Figure 3: Example of most important image patches" width="700"/><br>
Figure 3: Examples of most predictive image patches which could be provided to clinicians (purple: higher risk, green: lower risk). <br><br>

## Credits

### This work is mainly based on

Deep CoxPH modeling
- [Katzman, J. et al.](https://doi.org/10.1186/s12874-018-0482-1) "DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network." BMC Med Res Methodol 18, 24 (2018).

The Sparse BagNet for classification
- [Djoumessi, K. et al.](https://openreview.net/forum?id=us8BFTsWOq)
 "Sparse activations for interpretable disease grading." Medical Imaging with Deep Learning (2023).

Baselines: SOTA end-to-end AMD progression models
- [Babenko, B. et al.](https://arxiv.org/abs/1904.05478) "Predicting progression of age-related macular degeneration from fundus images using deep learning." arXiv preprint (2019).
- [Yan, Q. et al.](https://doi.org/10.1038/s42256-020-0154-9) "Deep-learning-based prediction of late age-related macular degeneration progression." Nat Mach Intell 2, 141–150 (2020). 
  - Note: We used the authors' model variant that does not rely on gene data and is trained end-to-end on fundus images.

### This work includes code adaptations from

- scikit-survival: We adapted the [Breslow estimator](https://github.com/sebp/scikit-survival/blob/master/sksurv/linear_model/coxph.py) to store the baseline hazard function and survival function and subsequently init the estimator from the saved data.
- auton-survival: We adapted the CoxPH loss implementation from the [authors' code](https://github.com/autonlab/auton-survival/blob/master/auton_survival/models/cph/dcph_utilities.py).
- [BagNet](https://github.com/wielandbrendel/bag-of-local-features-models) (Brendel and Bethge, 2019) and [Sparse BagNet](https://github.com/kdjoumessi/interpretable-sparse-activation) (Djoumessi et al., 2023): We adapted the BagNet implementations to survival modelling.
- Huang et al., 2023: We adapted the data augmentation from the [authors' code](https://github.com/YijinHuang/pytorch-classification/blob/master/data/transforms.py).

## Cite
<!-- Until the MICCAI 2024 proceedings are published, please cite this preprint: 
```bibtex
@inproceedings{gervelmeyer2024interpretable,
	author = {Gervelmeyer, Julius and Müller, Sarah and Djoumessi, Kerol and Merle, David and Clark, Simon J and Koch, Lisa and Berens, Philipp},
	title = {Interpretable-by-design Deep Survival Analysis for Disease Progression Modeling},
	year = {2024},
	doi = {10.1101/2024.07.11.24310270},
	URL = {https://www.medrxiv.org/content/early/2024/07/11/2024.07.11.24310270},
	booktitle = {Accepted at Medical Image Computing and Computer Assisted Intervention – MICCAI 2024}
}
``` -->

```bibtex
@InProceedings{gervelmeyer2024interpretable,
        author = { Gervelmeyer, Julius and Müller, Sarah and Djoumessi, Kerol and Merle, David and Clark, Simon J. and Koch, Lisa and Berens, Philipp},
        title = { { Interpretable-by-design Deep Survival Analysis for Disease Progression Modeling } },
        booktitle = {Proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024},
        year = {2024},
        publisher = {Springer Nature Switzerland},
        volume = {LNCS 15010},
        month = {October},
	URL = {https://papers.miccai.org/miccai-2024/421-Paper1325.html}
}
```


<br>

----------
[Back to top](#interpretable-by-design-deep-survival-analysis-for-disease-progression-modeling)
<br>
<br>
