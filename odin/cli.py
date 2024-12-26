import base64
import json
import logging
import os
import random
import shutil
import subprocess
import uuid
import click
from colorama import Fore
import cv2
import yaml
import albumentations as A

from odin.chronicle_utils import get_chronicle_name
from odin.dataset_classification import DatasetCommandsClassification
from odin.project_utils import get_project_info, get_project_pretty_type

logging.basicConfig(
    level=logging.INFO, format=f"[{Fore.YELLOW}%(asctime)s{Fore.RESET}] %(message)s"
)

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


@click.group()
def cli():
    pass


@click.command("start")
@click.option(
    "--type",
    default="detection",
    help="the project's type, must be either 'detection' (object detection) or 'classification'.",
)
@click.argument("project_name")
def start(type, project_name):
    """Starts a new machine vision project."""
    project_general_info = {"name": project_name, "type": type, "version": "0.1.0"}

    logging.info(
        f"Creating project structure for {Fore.CYAN}{project_name}{Fore.RESET}, a {Fore.CYAN}{get_project_pretty_type(type)}{Fore.RESET} project."
    )

    try:
        data = open("project.yaml", "r", encoding="utf8").read()
        if len(data) > 0:
            logging.info(
                f"There is already a project created in this location. Use {Fore.CYAN}odin wrath{Fore.RESET} to delete the project, then run the command again."
            )
            return
    except:
        pass

    with open("project.yaml", "w", encoding="utf8") as wf:
        wf.write(yaml.dump(project_general_info))

    # Datasets
    dataset_parent = f"{os.path.abspath('.')}\\datasets"
    if not os.path.exists(dataset_parent):
        os.makedirs(dataset_parent)

        logging.info(f"Succesfully created {Fore.CYAN}datasets{Fore.RESET}")

    logging.info(f"Creating {Fore.CYAN}datasets examples{Fore.RESET}...")

    if type == "classification":
        dataset_example_classif = f"{os.path.abspath('.')}\\datasets\\dataset_1_classif"
        if not os.path.exists(dataset_example_classif):
            os.makedirs(dataset_example_classif)

            os.makedirs(f"{os.path.abspath('.')}\\datasets\\dataset_1_classif\\class_1")
            os.makedirs(f"{os.path.abspath('.')}\\datasets\\dataset_1_classif\\class_2")

            logging.info(
                f"Succesfully created dataset example at: {Fore.CYAN}datasets/dataset_1_classif{Fore.RESET}"
            )
    elif type == "detection":
        dataset_example_obd = f"{os.path.abspath('.')}\\datasets\\dataset_1_obd"
        if not os.path.exists(dataset_example_obd):
            os.makedirs(dataset_example_obd)

            os.makedirs(f"{os.path.abspath('.')}\\datasets\\dataset_1_obd\\train")
            os.makedirs(f"{os.path.abspath('.')}\\datasets\\dataset_1_obd\\val")
            os.makedirs(
                f"{os.path.abspath('.')}\\datasets\\dataset_1_obd\\train\\images"
            )
            os.makedirs(f"{os.path.abspath('.')}\\datasets\\dataset_1_obd\\val\\images")
            os.makedirs(
                f"{os.path.abspath('.')}\\datasets\\dataset_1_obd\\train\\labels"
            )
            os.makedirs(f"{os.path.abspath('.')}\\datasets\\dataset_1_obd\\val\\labels")

            with open(
                f"{os.path.abspath('.')}\\datasets\\dataset_1_obd\\data.yaml", "w"
            ) as wf:
                wf.write(EXAMPLE_YAML)

            logging.info(
                f"Succesfully created dataset example at: {Fore.CYAN}datasets/dataset_1_obd{Fore.RESET}"
            )

    # Chronicles
    # chronicles/chronicle-nick/run-uuid
    chronicles_parent = f"{os.path.abspath('.')}\\chronicles"
    if not os.path.exists(chronicles_parent):
        os.makedirs(chronicles_parent)

        with open(f"{chronicles_parent}\\README.md", "w", encoding="utf8") as wf:
            wf.write(README_CHRONICLES)

        logging.info(f"Succesfully created {Fore.CYAN}chronicles{Fore.RESET}")

    # Weights
    weights_parent = f"{os.path.abspath('.')}\\weights"
    if not os.path.exists(weights_parent):
        os.makedirs(weights_parent)

        with open(f"{weights_parent}\\README.md", "w", encoding="utf8") as wf:
            wf.write(README_WEIGHTS)

        logging.info(f"Succesfully created {Fore.CYAN}weights{Fore.RESET}")


@click.command("train")
@click.option(
    "--epochs",
    default=30,
    help="the number of epochs to train the model on, good values may differ from dataset to dataset.",
)
@click.option(
    "--device",
    default="cpu",
    help="the device to use to train the model, must be either 'cpu' or 'gpu'.",
)
@click.option(
    "--base_model",
    default="yolo11n.pt",
    help="the pre-trained model to use for training.",
)
@click.argument("dataset_name")
@click.argument(
    "chronicle_name",
    default=get_chronicle_name,
)
def train(epochs, device, base_model, dataset_name, chronicle_name):
    """Trains the model, generating a new chronile based on a specific dataset. The name of the chronicle is not required, but can be passed."""
    chronicle_info = {
        "name": chronicle_name,
        "dataset": dataset_name,
        "epochs": epochs,
        "device": device,
    }

    project_info = get_project_info()
    project_type = project_info["type"]
    task = "detect" if project_type == "detection" else "classify"

    logging.info("Starting training...")

    subprocess.run(
        [
            "yolo",
            task,
            "train",
            f"data={os.path.abspath('.')}\\datasets\\{dataset_name}\\data.yaml",
            f"epochs={epochs}",
            f"batch={'-1' if device == 'gpu' else '2'}",
            f"model={base_model}",
            "amp=false",
            "patience=10",
            "save_period=5",
            f"device={'0' if device == 'gpu' else 'cpu'}",
            f"project={chronicle_name}",
            f"name={str(uuid.uuid4())}",
            "exist_ok=true",
            "plots=true",
        ]
    )


@click.command("dataset")
@click.argument("action")
@click.argument("dataset_name")
@click.option(
    "-t",
    "--train",
    "train",
    default=70,
    help="represents the split of the training dataset",
)
@click.option(
    "-v",
    "--val",
    "val",
    default=30,
    help="represents the split of the validation dataset",
)
@click.option(
    "-a",
    "--augmentations",
    "augs",
    default=2,
    help="the amount of augmentations per image on augmentate command.",
)
@click.option(
    "-V",
    "--version",
    "rollver",
    default=None,
    help="the version to rollback to, must be a valid version.",
)
def dataset(action, dataset_name, train, val, augs, rollver):
    project_info = get_project_info()
    project_type = project_info["type"]

    if project_type == "detection":
        pass
    elif project_type == "classification":
        interpreter = DatasetCommandsClassification(dataset_name)

    def delete_dataset():
        dataset_path = f"{os.path.abspath('.')}\\datasets\\{dataset_name}"

        if not os.path.exists(dataset_path):
            logging.info(
                f"The dataset mentioned doesn't exist, create it by using the command {Fore.CYAN}odin dataset create {dataset_name}{Fore.RESET}."
            )
        else:
            if click.confirm(
                f"Do you want to continue? All your data will be lost (and you will bring {Fore.RED}The Ragnarok{Fore.RESET} to thy dataset!)"
            ):
                logging.info(
                    f"Aplying {Fore.RED}Ragnarok{Fore.RESET} to {Fore.CYAN}{dataset_name}{Fore.RESET}!"
                )

                try:
                    shutil.rmtree(f"{dataset_path}")

                    logging.info(
                        f"Successfully deleted {Fore.CYAN}{dataset_name}{Fore.RESET}."
                    )
                except:
                    logging.info(
                        f"{Fore.CYAN}Odin{Fore.RESET} was unable to delete {Fore.CYAN}{dataset_name}{Fore.RESET}."
                    )

    def create_detection():
        # Creating dataset
        logging.info(f"Creating dataset '{Fore.BLUE}{dataset_name}{Fore.RESET}'...")

        dataset_path = f"{os.path.abspath('.')}\\datasets\\{dataset_name}"

        if not os.path.exists(dataset_path):
            # Creating dataset final folder
            logging.info("Creating final folders...")

            dataset_info = {"type": "obd", "version": "0.1.0", "changes": {}}

            with open(f"{dataset_path}\\dataset.json", "w", encoding="utf8") as wf:
                wf.write(json.dumps(dataset_info))

            with open(
                f"{dataset_path}\\CUSTOM_DATASETS.md", "w", encoding="utf8"
            ) as wf:
                wf.write(README_CUSTOM_DATASETS)

            with open(f"{dataset_path}\\classes.txt", "w", encoding="utf8") as wf:
                wf.write("0: object")

            os.makedirs(dataset_path)
            os.makedirs(dataset_path + "\\train")
            os.makedirs(dataset_path + "\\train\\images")
            os.makedirs(dataset_path + "\\train\\labels")
            os.makedirs(dataset_path + "\\val")
            os.makedirs(dataset_path + "\\val\\images")
            os.makedirs(dataset_path + "\\val\\labels")

            logging.info("Succesfully created final folders.")

            # Creating dataset staging folder

            logging.info("Creating staging folders...")

            os.makedirs(dataset_path + "\\staging")
            os.makedirs(dataset_path + "\\staging\\images")
            os.makedirs(dataset_path + "\\staging\\labels")

            logging.info("Succesfully created staging folders.")

            # Praise the gods! Your dataset folders are created, you can now insert your images and labels on YOLO format at 'dataset_path\\staging' and then run 'odin dataset stage {dataset_name} --train=70 --val=30'
            logging.info(
                f"Praise the gods! Your dataset folders are created, you can now insert your {Fore.CYAN}images{Fore.RESET} and {Fore.CYAN}labels{Fore.RESET} on {Fore.BLUE}YOLO{Fore.RESET} format at {Fore.CYAN}{dataset_path}\\staging{Fore.RESET} and then run {Fore.CYAN}odin dataset stage {dataset_name} --train=70 --val=30{Fore.RESET} (tip: you can change the {Fore.CYAN}--train{Fore.RESET} and {Fore.CYAN}--val{Fore.RESET} values to increase or decrease the split of the dataset)."
            )

    def publish_detection():
        logging.info(f"Publishing dataset '{Fore.BLUE}{dataset_name}{Fore.RESET}'...")

        dataset_path = f"{os.path.abspath('.')}\\datasets\\{dataset_name}"

        if not os.path.exists(dataset_path):
            logging.info(
                f"The dataset mentioned doesn't exist, create it by using the command {Fore.CYAN}odin dataset create {dataset_name}{Fore.RESET}."
            )
        else:
            if (
                sum(
                    len(files)
                    for _, _, files in os.walk(f"{dataset_path}\\staging\\images")
                )
                == 0
            ):
                logging.info(
                    f"The {Fore.CYAN}Staging{Fore.RESET} dataset is empty, so nothing will be published or updated."
                )
                return

            logging.info("Publishing with the following splits:")
            logging.info(f"{Fore.CYAN}train{Fore.RESET}: {train}%")
            logging.info(f"{Fore.CYAN}val{Fore.RESET}: {val}%")

            count_train = int(
                (train / 100)
                * sum(
                    len(files)
                    for _, _, files in os.walk(f"{dataset_path}\\staging\\images")
                )
            )

            count_val = int(
                (val / 100)
                * sum(
                    len(files)
                    for _, _, files in os.walk(f"{dataset_path}\\staging\\images")
                )
            )

            def publish_data(destination, max_file):
                images = []
                labels = []

                for _, _, files in os.walk(f"{dataset_path}\\staging\\images"):
                    images = files
                    break

                for _, _, files in os.walk(f"{dataset_path}\\staging\\labels"):
                    labels = files
                    break

                for i in range(0, max_file + 1):
                    try:
                        image_stage_path = (
                            f"{dataset_path}\\staging\\images\\{images[i]}"
                        )
                        image_publish_path = (
                            f"{dataset_path}\\{destination}\\images\\{images[i]}"
                        )

                        label_stage_path = (
                            f"{dataset_path}\\staging\\labels\\{labels[i]}"
                        )
                        label_publish_path = (
                            f"{dataset_path}\\{destination}\\labels\\{labels[i]}"
                        )

                        shutil.move(
                            image_stage_path,
                            image_publish_path,
                        )

                        shutil.move(
                            label_stage_path,
                            label_publish_path,
                        )
                    except IndexError:
                        pass

            publish_data("train", count_train)
            logging.info(f"Succesfully published {Fore.GREEN}train{Fore.RESET} data")
            publish_data("val", count_val)
            logging.info(f"Succesfully published {Fore.GREEN}val{Fore.RESET} data")

    def status_detection():
        dataset_path = f"{os.path.abspath('.')}\\datasets\\{dataset_name}"

        if not os.path.exists(dataset_path):
            logging.info(
                f"The dataset mentioned doesn't exist, create it by using the command {Fore.CYAN}odin dataset create {dataset_name}{Fore.RESET}."
            )
        else:
            # Status
            status = {
                "empty": f"{Fore.YELLOW}Empty{Fore.RESET}",
                "staging": f"{Fore.YELLOW}Staging{Fore.RESET}",
                "published": f"{Fore.GREEN}Published{Fore.RESET}",
                "published_staging": f"{Fore.CYAN}Published and Staging{Fore.RESET}",
            }

            staging = sum(
                len(files)
                for _, _, files in os.walk(f"{dataset_path}\\staging\\images")
            )
            train = sum(
                len(files) for _, _, files in os.walk(f"{dataset_path}\\train\\images")
            )

            if staging > 0 and train > 0:
                logging.info(
                    f"Retrieving {Fore.CYAN}{dataset_name}{Fore.RESET} status..."
                )
                logging.info(f"Status: {status['published_staging']}")
            elif staging > 0 and train == 0:
                logging.info(
                    f"Retrieving {Fore.CYAN}{dataset_name}{Fore.RESET} status..."
                )
                logging.info(f"Status: {status['staging']}")
            elif train > 0 and staging == 0:
                logging.info(
                    f"Retrieving {Fore.CYAN}{dataset_name}{Fore.RESET} status..."
                )
                logging.info(f"Status: {status['published']}")
            else:
                logging.info(
                    f"Retrieving {Fore.CYAN}{dataset_name}{Fore.RESET} status..."
                )
                logging.info(f"Status: {status['empty']}")

            # Image Count
            train_images_count = sum(
                len(files) for _, _, files in os.walk(f"{dataset_path}\\train\\images")
            )
            val_images_count = sum(
                len(files) for _, _, files in os.walk(f"{dataset_path}\\val\\images")
            )

            logging.info(
                f"Images on {Fore.CYAN}train{Fore.RESET}: {train_images_count}"
            )
            logging.info(f"Images on {Fore.CYAN}val{Fore.RESET}: {val_images_count}")

            # Class count
            if train == 0:
                return

            with open(f"{dataset_path}\\data.yaml", "r", encoding="utf8") as f:
                dataset_yaml = yaml.safe_load(f)

            train_labels = []
            val_labels = []
            for _, _, files in os.walk(f"{dataset_path}\\train\\labels"):
                train_labels = files
            for _, _, files in os.walk(f"{dataset_path}\\val\\labels"):
                val_labels = files

            train_label_count = {}
            val_label_count = {}

            for label_file in train_labels:
                with open(
                    f"{dataset_path}\\train\\labels\\{label_file}", "r", encoding="utf8"
                ) as f:
                    data = f.read().split("\n")

                    for line in data:
                        class_id = line.split(" ")[0]
                        try:
                            train_label_count[class_id] += 1
                        except:
                            train_label_count[class_id] = 1

            for label_file in val_labels:
                with open(
                    f"{dataset_path}\\val\\labels\\{label_file}", "r", encoding="utf8"
                ) as f:
                    data = f.read().split("\n")

                    for line in data:
                        class_id = line.split(" ")[0]
                        try:
                            val_label_count[class_id] += 1
                        except:
                            val_label_count[class_id] = 1

            logging.info(f"Class count on {Fore.CYAN}train{Fore.RESET}:")
            for class_id in train_label_count:
                try:
                    logging.info(
                        f"{dataset_yaml['names'][int(class_id)]}: {train_label_count[class_id]}"
                    )
                except:
                    pass

            logging.info(f"Class count on {Fore.CYAN}val{Fore.RESET}:")
            for class_id in val_label_count:
                try:
                    logging.info(
                        f"{dataset_yaml['names'][int(class_id)]}: {val_label_count[class_id]}"
                    )
                except:
                    pass

    def augmentate_detection():
        dataset_path = f"{os.path.abspath('.')}\\datasets\\{dataset_name}"

        if not os.path.exists(dataset_path):
            logging.info(
                f"The dataset mentioned doesn't exist, create it by using the command {Fore.CYAN}odin dataset create {dataset_name}{Fore.RESET}."
            )
        else:
            try:
                try:
                    with open(
                        f"{dataset_path}\\data.yaml", "r", encoding="utf8"
                    ) as data:
                        data_loaded = yaml.safe_load(data)
                except:
                    logging.info(
                        f"The dataset's {Fore.CYAN}data.yaml{Fore.RESET} wasn't found. Create one by running the command {Fore.CYAN}odin dataset yaml {dataset_name}{Fore.RESET}"
                    )
                    return

                images = []
                for _, _, files in os.walk(f"{dataset_path}\\train\\images"):
                    images = files
                    break

                logging.info(
                    f"Augmentating {Fore.CYAN}{len(images)}{Fore.RESET} images to a total of {Fore.CYAN}{len(images)+(len(images)*augs)}{Fore.RESET} images..."
                )
                for image_file in images:
                    image = cv2.imread(f"{dataset_path}\\train\\images\\{image_file}")

                    image_id = (
                        image_file.split(".png")[0].split(".jpg")[0].split(".jpeg")[0]
                    )

                    image_height, image_width, image_channels = image.shape
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    bboxes = list(
                        map(
                            lambda y: (
                                [
                                    int((float(y[1]) - float(y[3]) / 2) * image_width),
                                    int((float(y[2]) - float(y[4]) / 2) * image_height),
                                    float(y[3]) * image_width,
                                    float(y[4]) * image_height,
                                ]
                                if len(y) > 1
                                else None
                            ),
                            list(
                                map(
                                    lambda x: x.split(" "),
                                    open(
                                        f"{dataset_path}\\train\\labels\\{image_id}.txt"
                                    )
                                    .read()
                                    .split("\n"),
                                )
                            ),
                        )
                    )
                    try:
                        bboxes.remove(None)
                    except:
                        pass
                    classes = list(
                        map(
                            lambda y: y[0] if len(y) > 1 else None,
                            list(
                                map(
                                    lambda x: x.split(" "),
                                    open(
                                        f"{dataset_path}\\train\\labels\\{image_id}.txt"
                                    )
                                    .read()
                                    .split("\n"),
                                )
                            ),
                        )
                    )
                    try:
                        classes.remove(None)
                    except:
                        pass
                    classes = list(map(lambda z: int(z), classes))  # type: ignore
                    classes_to_name = data_loaded["names"]

                    transform = A.Compose(
                        [
                            A.HorizontalFlip(p=0.5),
                            # A.ShiftScaleRotate(p=0.5),
                            A.RandomCropFromBorders(p=0.5),
                            A.RandomBrightnessContrast(p=0.5),
                            A.RGBShift(
                                r_shift_limit=(-50, 50),
                                g_shift_limit=(-50, 50),
                                b_shift_limit=(-50, 50),
                                p=0.5,
                            ),
                            A.CLAHE(p=0.5),
                            A.ISONoise(intensity=(0.3, 0.5), p=0.5),
                            A.RandomGravel(p=0.5),
                            A.HueSaturationValue(p=0.5),
                        ],
                        bbox_params=A.BboxParams(
                            format="coco", label_fields=["classes"]
                        ),
                    )

                    random.seed(7)
                    data_to_save = []

                    for i in range(0, augs):
                        data_to_save.append(
                            transform(image=image, bboxes=bboxes, classes=classes)
                        )

                    def get_yolo_bboxes(bboxes):
                        final_bboxes = []
                        for bbox in bboxes:
                            x_min, y_min, w, h = bbox

                            x_center = (x_min + w / 2) / image_width
                            y_center = (y_min + h / 2) / image_height
                            width = w / image_width
                            height = h / image_height

                            final_bboxes.append([x_center, y_center, width, height])
                        return final_bboxes

                    def get_yolo_label(classes, bboxes):
                        lines = []
                        for i in range(0, len(classes)):
                            bbox = list(map(lambda x: str(x), bboxes[i]))
                            bboxes_lines = " ".join(bbox)
                            lines.append(f"{classes[i]} {bboxes_lines}")
                        return "\n".join(lines)

                    annotation_id = 0
                    for data in data_to_save:
                        image = cv2.cvtColor(data["image"], cv2.COLOR_BGR2RGB)
                        cv2.imwrite(
                            f"{dataset_path}\\train\\images\\{image_id}-{annotation_id}.png",
                            image,
                        )

                        bboxes = get_yolo_bboxes(data["bboxes"])
                        classes = list(map(lambda x: int(x), data["classes"]))
                        yolo_label = get_yolo_label(classes, bboxes)

                        with open(
                            f"{dataset_path}\\train\\labels\\{image_id}-{annotation_id}.txt",
                            "w",
                        ) as wf:
                            wf.write(yolo_label)

                        annotation_id += 1

                logging.info("Succesfully augmented all images.")
            except Exception as e:
                print(e)
                pass

    def yaml_detection():
        dataset_path = f"{os.path.abspath('.')}\\datasets\\{dataset_name}"

        if not os.path.exists(dataset_path):
            logging.info(
                f"The dataset mentioned doesn't exist, create it by using the command {Fore.CYAN}odin dataset create {dataset_name}{Fore.RESET}."
            )
        else:
            try:
                with open(f"{dataset_path}\\classes.txt", "r", encoding="utf8") as f:
                    data = list(
                        map(
                            lambda x: x.split(":")[-1].replace(" ", ""),
                            f.read().split("\n"),
                        )
                    )

                dataset_yaml = {
                    "path": dataset_path,
                    "train": "train\\images",
                    "val": "val\\images",
                    "names": {},
                }

                for class_id in range(0, len(data)):
                    dataset_yaml["names"][class_id] = data[class_id]

                with open(f"{dataset_path}\\data.yaml", "w", encoding="utf8") as wf:
                    wf.write(yaml.dump(dataset_yaml))

                logging.info(
                    f"Succesfully generated {Fore.CYAN}data.yaml{Fore.RESET} for {Fore.CYAN}{dataset_name}{Fore.RESET}"
                )
            except:
                logging.info(
                    f"Your {Fore.CYAN}classes.txt{Fore.RESET} is either empty or non-existant. If you don't have a {Fore.CYAN}classes.txt{Fore.RESET} in your dataset, please provide one so {Fore.CYAN}Odin{Fore.RESET} can generate the YAML file. Read the documentation to know more about how it should be at {Fore.CYAN}datasets/{dataset_name}/CUSTOM_DATASETS.md{Fore.RESET}."
                )

    if action == "create":
        if project_type == "detection":
            create_detection()
        elif project_type == "classification":
            interpreter.create()
    elif action == "publish":
        if project_type == "detection":
            publish_detection()
        elif project_type == "classification":
            interpreter.publish(train, val)
    elif action == "status":
        if project_type == "detection":
            status_detection()
        elif project_type == "classification":
            interpreter.status()
    elif action == "delete":
        delete_dataset()
    elif action == "augmentate":
        if project_type == "detection":
            augmentate_detection()
        elif project_type == "classification":
            interpreter.augmentate(augs)
    elif action == "yaml":
        if project_type == "detection":
            yaml_detection()
        else:
            logging.info(
                f"Your dataset type is defined as {Fore.CYAN}Classification{Fore.RESET}, this type of project doesn't require a {Fore.CYAN}data.yaml{Fore.RESET}."
            )
    elif action == "rollback":
        if project_type == "detection":
            pass
        elif project_type == "classification":
            interpreter.rollback(rollver)


@click.command("wrath")
def wrath():
    logging.info(
        f"Warning! Beyond this decision rests the {Fore.RED}DOOM{Fore.RESET} of all your {Fore.CYAN}datasets{Fore.RESET}, {Fore.CYAN}chronicles{Fore.RESET} and {Fore.CYAN}weights{Fore.RESET}."
    )

    if click.confirm("Do you want to continue?"):
        logging.info(
            f"Laying down {Fore.CYAN}Odin's{Fore.RESET} wrath against thy foe(der)s!"
        )

        try:
            shutil.rmtree("chronicles")
        except:
            pass

        try:
            shutil.rmtree("datasets")
        except:
            pass

        try:
            shutil.rmtree("weights")
        except:
            pass

        try:
            os.remove("project.yaml")
        except:
            pass

        logging.info("This land has been purged.")
    else:
        logging.info(f"So the {Fore.RED}Ragnarok{Fore.RESET} must wait.")


cli.add_command(start)
cli.add_command(train)
cli.add_command(wrath)
cli.add_command(dataset)

if __name__ == "__main__":
    cli()
