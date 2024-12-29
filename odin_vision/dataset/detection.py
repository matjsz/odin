

import base64
import json
import logging
import os
import random
import re
import shutil

import albumentations as A

from colorama import Fore
import cv2
import yaml
from odin_vision.dataset.base import BaseDatasetCommands
from odin_vision.constants import README_CUSTOM_DATASETS


class DatasetCommandsDetection(BaseDatasetCommands):
    def __init__(self, dataset_name):
        self.type = "detection"
        self.dataset_name = dataset_name
        self.dataset_path = f"{os.path.abspath('.')}\\datasets\\{dataset_name}"
        
    def _create_dataset_metadata_files(self, **kwargs):
        self._try_create_folder(self.dataset_path)
        
        dataset_info = {"type": "obd", "version": "0.1.0", "snapshots": {}}
        
        with open(f"{self.dataset_path}\\dataset.json", "w", encoding="utf8") as wf:
            wf.write(json.dumps(dataset_info))

        with open(f"{self.dataset_path}\\snapshot.json", "w", encoding="utf8") as wf:
            wf.write(json.dumps({}))
            
        with open(
            f"{self.dataset_path}\\CUSTOM_DATASETS.md", "w", encoding="utf8"
        ) as wf:
            wf.write(README_CUSTOM_DATASETS)
            
        with open(f"{self.dataset_path}\\classes.txt", "w", encoding="utf8") as wf:
            wf.write("0: object")
            
        logging.info("Creating staging folders...")
        
        self._try_create_folder(self.dataset_path + "\\staging")
        self._try_create_folder(self.dataset_path + "\\staging\\images")
        self._try_create_folder(self.dataset_path + "\\staging\\labels")

        logging.info("Succesfully created staging folders.")

        self._try_create_folder(self.dataset_path)
        self._try_create_folder(self.dataset_path + "\\train")
        self._try_create_folder(self.dataset_path + "\\train\\images")
        self._try_create_folder(self.dataset_path + "\\train\\labels")
        self._try_create_folder(self.dataset_path + "\\val")
        self._try_create_folder(self.dataset_path + "\\val\\images")
        self._try_create_folder(self.dataset_path + "\\val\\labels")
        
        logging.info("Succesfully created dataset folders.")
        
        logging.info(
            f"Praise the gods! Your dataset folders are created, you can now insert your {Fore.CYAN}images{Fore.RESET} and {Fore.CYAN}labels{Fore.RESET} on {Fore.BLUE}YOLO{Fore.RESET} format at {Fore.CYAN}{self.dataset_path}\\staging{Fore.RESET} and then run {Fore.CYAN}odin dataset stage {self.dataset_name} --train=70 --val=30{Fore.RESET} (tip: you can change the {Fore.CYAN}--train{Fore.RESET} and {Fore.CYAN}--val{Fore.RESET} values to increase or decrease the split of the dataset)."
        )
        
    def _get_label_name_from_image_name(self, image_name="", **kwargs):
        pattern = r"\b(?:jpg|jpeg|png|webp|gif|bmp|tiff|svg)\b"
        return re.sub(pattern, "ext", image_name, flags=re.IGNORECASE)
        
    def _get_label_binary(self, label_path="", **kwargs):
        with open(label_path, "rb") as f:
            return f.read()
        
    def _add_artifact_to_version_snapshot(
        self, snapshot={}, dataset_split="", label_name="", label_binary="", image_name="", image_binary="", **kwargs
    ):
        try:
            snapshot[dataset_split]['images'].append(
                {
                    "filename": image_name,
                    "binary": base64.b64encode(image_binary).decode("utf8"),
                }
            )
        except:
            snapshot[dataset_split]['images'] = [
                {
                    "filename": image_name,
                    "binary": base64.b64encode(image_binary).decode("utf8"),
                }
            ]
            
        try:
            snapshot[dataset_split]['labels'].append(
                {
                    "filename": label_name,
                    "binary": base64.b64encode(label_binary).decode("utf8"),
                }
            )
        except:
            snapshot[dataset_split]['labels'] = [
                {
                    "filename": label_name,
                    "binary": base64.b64encode(label_binary).decode("utf8"),
                }
            ]
            
    def _execute_data_publishment(self, snapshot={}, split="", split_max_file=0, **kwargs):
        images = []
        labels = []

        for _, _, files in os.walk(f"{self.dataset_path}\\staging\\images"):
            images = files
            break

        for _, _, files in os.walk(f"{self.dataset_path}\\staging\\labels"):
            labels = files
            break
        
        for i in range(0, split_max_file + 1):
            try:
                image_stage_path = (
                    f"{self.dataset_path}\\staging\\images\\{images[i]}"
                )
                image_publish_path = (
                    f"{self.dataset_path}\\{split}\\images\\{images[i]}"
                )

                label_stage_path = (
                    f"{self.dataset_path}\\staging\\labels\\{labels[i]}"
                )
                label_publish_path = (
                    f"{self.dataset_path}\\{split}\\labels\\{labels[i]}"
                )

                shutil.move(
                    image_stage_path,
                    image_publish_path,
                )

                shutil.move(
                    label_stage_path,
                    label_publish_path,
                )
                
                image_binary = open(image_publish_path, "rb").read()
                label_binary = open(label_publish_path, "rb").read()
                
                self._add_artifact_to_version_snapshot(snapshot, split, labels[i], label_binary, images[i], image_binary)
            except IndexError:
                pass
            
    def _publish_data(self, update_type="", train=0, val=0, **kwargs):
        if (
            sum(
                len(files)
                for _, _, files in os.walk(f"{self.dataset_path}\\staging\\images")
            )
            == 0
        ):
            logging.info(
                f"The {Fore.CYAN}Staging{Fore.RESET} dataset is empty, so nothing will be published or updated."
            )
            return
        
        base_version = json.loads(
            open(f"{self.dataset_path}\\dataset.json", "r", encoding="utf8").read()
        )["version"]
        temp_version = self._upgrade_version(base_version, update_type)
        
        snapshot = {
            "staging": {},
            "train": {},
            "val": {},
        }
            
        logging.info("Publishing with the following splits:")
        logging.info(f"{Fore.CYAN}train{Fore.RESET}: {train}%")
        logging.info(f"{Fore.CYAN}val{Fore.RESET}: {val}%")
        
        count_train = int(
            (train / 100)
            * sum(
                len(files)
                for _, _, files in os.walk(f"{self.dataset_path}\\staging\\images")
            )
        )

        count_val = int(
            (val / 100)
            * sum(
                len(files)
                for _, _, files in os.walk(f"{self.dataset_path}\\staging\\images")
            )
        )
        
        self._execute_data_publishment(snapshot, "train", count_train)
        logging.info(
            f"Succesfully published {Fore.GREEN}train{Fore.RESET} data."
        )
        self._execute_data_publishment(snapshot, "val", count_val)
        logging.info(
            f"Succesfully published {Fore.GREEN}val{Fore.RESET} data."
        )
        
        dataset_info = json.loads(
            open(f"{self.dataset_path}\\dataset.json", "r", encoding="utf8").read()
        )
        snapshot_info = json.loads(
            open(f"{self.dataset_path}\\snapshot.json", "r", encoding="utf8").read()
        )

        snapshot_info[temp_version] = snapshot
        dataset_info["version"] = temp_version

        with open(f"{self.dataset_path}\\dataset.json", "w", encoding="utf8") as wf:
            wf.write(json.dumps(dataset_info, indent=4))
        logging.info(
            f"Succesfully updated dataset version to {Fore.CYAN}v{temp_version}{Fore.RESET}"
        )

        with open(f"{self.dataset_path}\\snapshot.json", "w", encoding="utf8") as wf:
            wf.write(json.dumps(snapshot_info, indent=4))
        logging.info(
            f"Succesfully registered snapshots for dataset {Fore.CYAN}v{temp_version}{Fore.RESET}"
        )
        
    def _status_sum_staging(self, **kwargs):
        return sum(
            len(files)
            for _, _, files in os.walk(
                f"{self.dataset_path}\\staging\\images"
            )
        )

    def _status_sum_train(self, **kwargs):
        return sum(
            len(files)
            for _, _, files in os.walk(
                f"{self.dataset_path}\\train\\images"
            )
        )
    
    def _status_sum_val(self, **kwargs):
        return sum(
            len(files)
            for _, _, files in os.walk(
                f"{self.dataset_path}\\val\\images"
            )
        )
        
    def _status_show_additional_info(self, **kwargs):
        train = self._status_sum_train()
        
        if train == 0:
            return

        with open(f"{self.dataset_path}\\data.yaml", "r", encoding="utf8") as f:
            dataset_yaml = yaml.safe_load(f)

        train_labels = []
        val_labels = []
        for _, _, files in os.walk(f"{self.dataset_path}\\train\\labels"):
            train_labels = files
        for _, _, files in os.walk(f"{self.dataset_path}\\val\\labels"):
            val_labels = files

        train_label_count = {}
        val_label_count = {}

        for label_file in train_labels:
            with open(
                f"{self.dataset_path}\\train\\labels\\{label_file}", "r", encoding="utf8"
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
                f"{self.dataset_path}\\val\\labels\\{label_file}", "r", encoding="utf8"
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
            
    def _augmentate_data(self, augmentation_amount=0, **kwargs):
        try:
            try:
                with open(
                    f"{self.dataset_path}\\data.yaml", "r", encoding="utf8"
                ) as data:
                    data_loaded = yaml.safe_load(data)
            except:
                logging.info(
                    f"The dataset's {Fore.CYAN}data.yaml{Fore.RESET} wasn't found. Create one by running the command {Fore.CYAN}odin dataset yaml {self.dataset_name}{Fore.RESET}"
                )
                return

            images = []
            for _, _, files in os.walk(f"{self.dataset_path}\\train\\images"):
                images = files
                break

            logging.info(
                f"Augmentating {Fore.CYAN}{len(images)}{Fore.RESET} images to a total of {Fore.CYAN}{len(images)+(len(images)*augmentation_amount)}{Fore.RESET} images..."
            )
            for image_file in images:
                image = cv2.imread(f"{self.dataset_path}\\train\\images\\{image_file}")

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
                                    f"{self.dataset_path}\\train\\labels\\{image_id}.txt"
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
                                    f"{self.dataset_path}\\train\\labels\\{image_id}.txt"
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

                for i in range(0, augmentation_amount):
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
                        f"{self.dataset_path}\\train\\images\\{image_id}-{annotation_id}.png",
                        image,
                    )

                    bboxes = get_yolo_bboxes(data["bboxes"])
                    classes = list(map(lambda x: int(x), data["classes"]))
                    yolo_label = get_yolo_label(classes, bboxes)

                    with open(
                        f"{self.dataset_path}\\train\\labels\\{image_id}-{annotation_id}.txt",
                        "w",
                    ) as wf:
                        wf.write(yolo_label)

                    annotation_id += 1

            logging.info("Succesfully augmented all images.")
        except Exception as e:
            print(e)
            pass

    def _rollback_dataset(self, snapshot_info={}, rollback_version="", dataset_folder="", split_folder="", **kwargs):
        shutil.rmtree(f"{self.dataset_path}\\{dataset_folder}\\{split_folder}\\images")
        os.makedirs(f"{self.dataset_path}\\{dataset_folder}\\{split_folder}\\images")
        shutil.rmtree(f"{self.dataset_path}\\{dataset_folder}\\{split_folder}\\labels")
        os.makedirs(f"{self.dataset_path}\\{dataset_folder}\\{split_folder}\\labels")    
        
        for file_data in snapshot_info[rollback_version][dataset_folder][split_folder]:
            file_bin = file_data["binary"]
            file_name = file_data["filename"]

            encoded_content = file_bin.encode()
            content = base64.b64decode(encoded_content)

            with open(
                f"{self.dataset_path}\\{dataset_folder}\\{split_folder}\\{file_name}",
                "wb",
            ) as wf:
                wf.write(content)
        
        logging.info(
            f"Succesfully executed rollback at {Fore.CYAN}{dataset_folder}{Fore.RESET}"
        )

    def yaml(self, **kwargs):
        dataset_path = f"{os.path.abspath('.')}\\datasets\\{self.dataset_name}"

        if not os.path.exists(dataset_path):
            logging.info(
                f"The dataset mentioned doesn't exist, create it by using the command {Fore.CYAN}odin dataset create {self.dataset_name}{Fore.RESET}."
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
                    f"Succesfully generated {Fore.CYAN}data.yaml{Fore.RESET} for {Fore.CYAN}{self.dataset_name}{Fore.RESET}"
                )
            except:
                logging.info(
                    f"Your {Fore.CYAN}classes.txt{Fore.RESET} is either empty or non-existant. If you don't have a {Fore.CYAN}classes.txt{Fore.RESET} in your dataset, please provide one so {Fore.CYAN}Odin{Fore.RESET} can generate the YAML file. Read the documentation to know more about how it should be at {Fore.CYAN}datasets/{self.dataset_name}/CUSTOM_DATASETS.md{Fore.RESET}."
                )
            