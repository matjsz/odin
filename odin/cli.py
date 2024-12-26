import logging
import os
import shutil
import subprocess
import uuid
import click
from colorama import Fore

from chronicle_utils import get_chronicle_name
from dataset_classification import DatasetCommandsClassification
from dataset_detection import DatasetCommandsDetection
from start import StartCommand
from project_utils import get_project_info

logging.basicConfig(
    level=logging.INFO, format=f"[{Fore.YELLOW}%(asctime)s{Fore.RESET}] %(message)s"
)

@click.group()
def cli():
    pass

@click.command("start")
@click.option(
    "-t",
    "--type",
    "project_type",
    default="detection",
    help="the project's type, must be either 'detection' (object detection) or 'classification'.",
)
@click.argument("project_name")
def start(project_type, project_name):
    """Starts a new machine vision project."""
    builder = StartCommand(project_type, project_name)
    
    project_builders = {
        "classification": builder.classification,
        "detection": builder.detection
    }
    
    builder.create_datasets_structure()
    
    project_builders[project_type]()

    builder.create_models_structure()

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
        interpreter = DatasetCommandsDetection(dataset_name)
    elif project_type == "classification":
        interpreter = DatasetCommandsClassification(dataset_name)
        
    actions = {
        "create": interpreter.create,
        "publish": interpreter.publish,
        "status": interpreter.status,
        "delete": interpreter.delete,
        "augmentate": interpreter.augmentate,
        "rollback": interpreter.rollback,
        "yaml": interpreter.yaml
    }
    
    actions[action](train=train, val=val, augmentation_amount=augs, rollver=rollver)

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
