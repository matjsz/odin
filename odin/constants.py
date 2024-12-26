import os


EXAMPLE_YAML = f"""
path: {os.path.abspath('.')}\\datasets\\dataset_1_odb
train: train\\images
val: val\\images

names:
    0: dog
    1: cat
"""

README_CHRONICLES = """
# Odin's Chronicles

Chronicles are essentially training sessions that contain runs. Each sessions uses a specific dataset, so a good practice is to use/create a new session for each new version of the project's dataset.

## Case Example

A simple example of a chronicle would be: 

```
{project_path}/chronicles/my-chronicle/run-id
# Which contains:
.../my-chronicle/run-id/weights/best.pt
.../my-chronicle/run-id/labels.jpg
.../my-chronicle/run-id/results.csv
.../my-chronicle/run-id/train_batch0.jpg
...
```
"""

README_WEIGHTS = """
# Odin's Weights

Weights are the final versions of a model training. This is nothing but a way to organize your project, in fact, the file stored here is the same file acquired at `chronicles/my-chronicle/run-id/weights`.

This folder is managed by Odin and should not be changed. The framework itself versions the weights.
"""

README_CUSTOM_DATASETS = """
# Custom Datasets on Odin

## Object Detection

All **Object Detection** datasets created by **Odin** have a **classes.txt** inside it, this file is extremely important if you don't know how to create a **data.yaml** (if you know, you can just create your own **data.yaml** without any problems, but it's advised to let **Odin** do it, so the path is not wrong).

### classes.txt

The file **classes.txt** should be in the following format:

```
0: dog
1: cat
2: person
3: chair
```

## Classification

"""
