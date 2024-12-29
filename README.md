<p align="center">
  <img src="https://raw.githubusercontent.com/matjsz/odin/refs/heads/main/docs/logo.png" style="">
</p>
<p align="center">
  <img src="https://img.shields.io/pypi/v/odin-vision.svg">
  <img src="https://img.shields.io/pypi/pyversions/odin-vision.svg">
</p>

# Odin

**Odin** is an open-source CLI framework for managing big computer vision projects that are based on YOLO.

## Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Initial setup](#initial-setup)
  - [Creating and managing your first dataset](#creating-and-managing-your-first-dataset)
  - [Training and maganing your first model](#training-and-maganing-your-first-model)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

# Installation

```
pip install odin-vision
```

## Features

Odin features includes not only, but mainly:

⚙️ Robust computer vision management tools, such as:
  - Dataset automated management (creation & version control)
  - Model automated management (training, testing & version control)
  - General project management with a robust project structure that can be automated to a pipeline.

📄 A detailed documentation is on the way!

## Usage

### Initial setup

1. Create a new folder with the **project's name**.

```
> mkdir my-project & cd my-project
```

2. Initialize your **Odin** project.

```
> odin start my-project --type detection
```

### Creating and managing your first dataset

1. Create your first dataset.

```
> odin dataset create --dataset_name my-dataset
```

2. Add your dataset data to the **Odin**-managed _my-dataset_.

Label a new dataset using [Label Studio](https://labelstud.io/), [Roboflow](https://roboflow.com/) or any other labelling tool. Once labelled, add your data to the **staging** folder inside your new dataset.

3. Publish your new dataset.

With your new dataset inside the **staging** folder, it's time to "publish" your dataset (publishing means officializing your dataset inside your project, nothing inside your **Odin** project will ever touch the web or get out of your environment).

```  
> odin dataset publish --dataset_name my-dataset
```

### Training and maganing your first model

1. Train a model on your published dataset.

```
> odin model train --dataset_name my-dataset
```

Once trained, your model will be available at **/chronicles/{chronicle_name}/weights/{model_name}**, but you should follow the good practices of the framework and just use the models inside **/weights** on the root of your project.

2. Test the model

```
> odin model test --chronicle_name {chronicle_name}
```

Once tested, you can publish the model if you want, it will really just add the model to the **weights** folder at the root of the project and properly version it, but it's a very good practice to have, it may maintain your project organized.

3. Publish your tested model

```
> odin model publish --chronicle_name {chronicle_name}
```

## Roadmap

Features planned for the next updates:

- [x] Automated dataset generation
- [x] Dataset versioning
- [x] Dataset rollback
- [x] Automated training
- [x] Model versioning
- [ ] Automated testing report generation
- [ ] Odin programmatic API
- [ ] Integration with PyTorch Lightning for performance

## Special thanks and Acknowledgments 

I'm very thankful for Ultralytics for disponibilizing the YOLO framework as an open-source project. This framework is used only in the training phase of the pipeline to, as it would be expected, train the models.

Special thanks for the OpenCV team for also maintaining the project alive and open-source, for the Ultralytics team as I've mentioned above, also the Albumentations team and of course, PyTorch, for providing the base for almost every deep neural network solution not just in this project, but most of the projects out there. tools like Odin wouldn't exist without these awesome solutions.

This project is powered by:
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [OpenCV Python](https://github.com/opencv/opencv-python)
- [Albumentations](https://github.com/albumentations-team/albumentations)
- [PyTorch](https://github.com/pytorch/pytorch)

## Contributing

If you find a bug, please open a [bug report](https://github.com/matjsz/odin/issues/new?labels=bug).
If you have an idea for an improvement or new feature, please open a [feature request](https://github.com/matjsz/odin/issues/new?labels=enhancement).