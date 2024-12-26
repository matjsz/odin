import logging
import os
import shutil
import subprocess
import uuid
import click
from colorama import Fore
import yaml

from odin.chronicle_utils import get_chronicle_name
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
    "--train", default=70, help="represents the split of the training dataset"
)
@click.option(
    "--val", default=30, help="represents the split of the validation dataset"
)
def dataset(action, dataset_name, train, val):
    if action == "create":
        # Creating dataset
        logging.info(f"Creating dataset '{Fore.BLUE}{dataset_name}{Fore.RESET}'...")

        dataset_path = f"{os.path.abspath('.')}\\datasets\\{dataset_name}"

        if not os.path.exists(dataset_path):
            # Creating dataset final folder
            logging.info("Creating final folders...")

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
    elif action == "stage":
        logging.info(f"Staging dataset '{Fore.BLUE}{dataset_name}{Fore.RESET}'...")

        dataset_path = f"{os.path.abspath('.')}\\datasets\\{dataset_name}"

        if not os.path.exists(dataset_path):
            logging.info(
                f"The dataset mentioned doesn't exist, create it by using the command {Fore.CYAN}odin dataset create {dataset_name}{Fore.RESET}."
            )
        else:
            logging.info("Staging with the following splits:")
            logging.info(f"{Fore.CYAN}train{Fore.RESET}: {train}")
            logging.info(f"{Fore.CYAN}train{Fore.RESET}: {train}")


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
