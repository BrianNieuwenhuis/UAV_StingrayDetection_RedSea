# UAV_StingrayDetection_RedSea
Code from: Nieuwenhuis, Brian O., Charlotte Turlier, Ioana-Andreea Ciocănaru, Benja Blaschke, Malika Kheireddine, Guido Leurs, Jesse E. M. Cochran, Laura L. Govers, and Burton H. Jones. 2026. "Fine-scale habitat partitioning of sympatric stingrays revealed by drone-based remote sensing and deep learning". https://doi.org/10.64898/2026.03.15.710512

This repository contains the following files:

	- best_hyperparameters_YOLOmodel.yaml: 
YAML file specifying the hyperparameters that generated the best results during 200 iterations of model tuning.

	- best_YOLOmodel.pt: 
YOLO model trained on our dataset (Annotated_YOLO_Dataset.zip) using the best hyperparameter setting observed during model tuning.

	- crop_annotated_images.py: 
Python script used to extract 1344 x 1344 crops of our annotations from the full-sized drone images.

	- evaluate_YOLOmodel.py: 
Python script used to test the YOLO model on our test set.

	- run_inference_YOLOmodel.py: 
Python script used to run inference on the sliced transect surveys with our best trained YOLO model.

	- slice_images.py: 
Python script to slice images into overlapping tiles for sliced inference.

	- train_default_YOLOmodel.py: 
Python script used to train a YOLO model on our dataset using un-tuned default parameters.

	- tune_YOLOmodel.py: 
Python script used to tune the model hyperparameters and image augmentation settings.

	- visualise_bounding_box.py: 
Python script used to inspect bounding box annotations or model predictions. Useful to double check the output of e.g. crop_annotated_images.py.


The YOLO model weights of our best model can be downloaded from here: https://github.com/BrianNieuwenhuis/UAV_StingrayDetection_RedSea/releases/download/v1.0.0/best_YOLOmodel.pt

