import os
import pprint
import cv2
import numpy as np
import yaml
from matplotlib import pyplot as plt

import albumentations as A

with open("data.yaml", "r", encoding="utf8") as data:
    data_loaded = yaml.safe_load(data)

BASE_DIR = os.path.abspath("train")

transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomCropFromBorders(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.CLAHE(p=0.5),
        A.ISONoise(intensity=(0.3, 0.5), p=0.5),
        A.RandomGravel(p=0.5),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["category_ids"]),
)


def augmentate_image(image_file, counter):
    image = cv2.imread(f"{BASE_DIR}\\images\\{image_file}")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    annotation_id, annotation_ext = os.path.splitext(image_file)

    if "annotated" in annotation_id:
        return

    label_file = annotation_id + ".txt"

    with open("data.yaml", "r") as file:
        dataset_data = yaml.safe_load(file)

    with open(f"train/labels/{label_file}") as f:
        individual_label_content = f.read().split("\n")

        bboxes = []
        for bbox_annotation in individual_label_content:
            temp_bbox = bbox_annotation.split(" ")
            bbox = []
            for value in temp_bbox:
                try:
                    bbox.append(float(value))
                except:
                    pass
            if len(bbox) >= 4:
                bboxes.append(bbox)

        # bboxes = list(
        #     map(
        #         lambda x: list(map(lambda y: float(y), x.split(" "))),
        #         individual_label_content,
        #     )
        # )

    class_ids = []

    for bbox in bboxes:
        try:
            class_id = bbox.pop(0)
            class_ids.append(int(class_id))
            # bbox.append(int(class_id))
        except:
            return

    pprint.pprint(class_ids)
    pprint.pprint(bboxes)

    # print(label_file)
    # pprint.pprint(bboxes)

    transformed = transform(image=image, bboxes=bboxes, category_ids=class_ids)

    transformed_image = transformed["image"]
    transformed_bboxes = transformed["bboxes"]

    label_file_content = ""
    for class_id, bbox in zip(class_ids, transformed_bboxes):
        iterable_bbox = list(map(lambda x: str(x), bbox))
        label_file_content += f"{class_id} {' '.join(iterable_bbox)}\n"

    cv2.imwrite(
        f"{BASE_DIR}\\augmented\\images\\{annotation_id}-annotated-{counter}{annotation_ext}",
        transformed_image,
    )
    with open(
        f"{BASE_DIR}\\augmented\\labels\\{annotation_id}-annotated-{counter}.txt", "w"
    ) as wf:
        wf.write(label_file_content)


for _, _, files in os.walk(f"{BASE_DIR}/images"):
    for file in files:
        try:
            for i in range(0, 2):
                augmentate_image(file, i)
        except Exception as e:
            print(e)

# augmentate_image("0ba5bf2b-frame_72000-0.png")
