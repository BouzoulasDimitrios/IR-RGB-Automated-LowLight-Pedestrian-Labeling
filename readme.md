# IR-RGB automated pedestrian labelling pipeline

This repository shows the steps depicted in the reasearch article on "**nighttime pedestrian detection**" submitted to [ITSC-2025](https://ieee-itsc.org/2025/).

# Introduction

The research concerns the process of automatically labelling pedestrians at night specifically aimed for RGB footage. The goal of the research was to provide and evaluate a pipeline that would proove that automated labelling can be performed on nighttime Infrred-RGB image pair for pedestrian detection and that it can yield results equal to those of ground truth labels. For this goal the following steps were taken:

1) **Automated Labelling**: Infrared images were labelled automatically using a fine tuned model for Infrared pedestrian detection.
2) **Label Tranfer**: The generated labels were transferred to the RGB couterparts of the infrared images.
3) **Evaluation**: Using the automatically labelled RGB images a nighttime pedestrian detection model is trained, similarly another model is trained using the ground truth labels provided in KAIST. We then compare and evaluate the two models on Sequence09 of KAIST, a sequqnce they have not trained on and a sequence that covers a wide range of scenarios 

<img src="./readme_resources/thesis_architecture.drawio.png" />



# Baseline model

For the baseline model of the research YOLOv11 was chosen. 
All of the steps described were performed with YOLOv11m, l and x to validate the results across a wider range of models.


# Dataset - KAIST

The dataset used for the research is the [KAIST dataset](https://soonminhwang.github.io/rgbt-ped-detection/). 
KAIST 



### Dataset Preprocessing

KAIST consists of 3 label classes `person`, `people` and `cyclist`. 

the `autolabeller_fine_tunning.ipynb` notebook goes through the fine tunning steps






















