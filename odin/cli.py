import logging
import os
import shutil
import subprocess
import uuid
import click
from colorama import Fore

from chronicle_utils import get_chronicle_name

logging.basicConfig(
    level=logging.INFO, format=f"[{Fore.CYAN}ODIN{Fore.RESET}][{Fore.YELLOW}%(asctime)s{Fore.RESET}] %(message)s"
)

@click.group()
def cli():
    pass

@click.command("start")
@click.argument("project_name", default=None, required=False)
@click.option(
    "-t",
    "--type",
    "project_type",
    default=None,
    help="the project's type, must be either 'detection' (object detection) or 'classification'.",
)
def start(project_type, project_name):
    """Starts a new machine vision project."""
    from start import StartCommand
    
    if not project_name:
        confirmed_name = False
        
        while not confirmed_name:
            project_name = click.prompt(
                f"What is the project's name?"
            )
            
            if click.confirm(f'{Fore.CYAN}{project_name}{Fore.RESET}, is this name right?'):
                confirmed_name = True
    
    if not project_type:
        project_type = click.prompt(
            f"What is the project's type?",
            show_choices=True,
            type=click.Choice([f'classification', f'detection']),
        )
    
    builder = StartCommand(project_type, project_name)
    
    project_builders = {
        "classification": builder.classification,
        "detection": builder.detection
    }
    
    builder.create_datasets_structure()
    
    project_builders[project_type]()

    builder.create_models_structure()

@click.command("model")
@click.argument("action", type=click.Choice([
    'train',
    'test',
    'publish'
]))
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
    default=None,
    help="the pre-trained model to use for training.",
)
@click.option(
    "--subset",
    default=100,
    help="the percentage of the training dataset to use for the actual training of the model.",
)
@click.option(
    "-C",
    "--chronicle",
    "chronicle_name",
    default=None,
    help="the name of the chronicle to use."
)
@click.option(
    "-D",
    "--dataset",
    "dataset_name",
    default=None,
    help="the name of the dataset to use."
)
def model(action, epochs, device, base_model, subset, dataset_name, chronicle_name):
    """Trains the model, generating a new chronile based on a specific dataset. The name of the chronicle is not required, but can be passed."""
    # Command interface for model training.
    
    from project_utils import get_project_info
    project_info = get_project_info()
    project_type = project_info["type"]
                
    if project_type == "detection":
        from training_detection import DetectionTrainingCommands
        interpreter = DetectionTrainingCommands(project_type, dataset_name)
    elif project_type == "classification":
        from training_classification import ClassificationTrainingCommands
        interpreter = ClassificationTrainingCommands(project_type, dataset_name)
    
    commands = {
        "train": interpreter.train,
        "test": interpreter.test,
        "publish": interpreter.publish
    }
    
    commands[action](epochs=epochs, project_name=project_info["name"], device=device, base_model=base_model, dataset_name=dataset_name, chronicle_name=chronicle_name, subset=subset)

@click.command("dataset")
@click.argument("action", type=click.Choice([
    'create',
    'publish',
    'status',
    'delete', 
    'augmentate',
    'rollback', 
    'yaml'
]))
@click.option("-D", "--dataset_name", "dataset_name", default=None, help="the dataset name.")
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
    """Command interface for dataset management."""
    
    from dataset_classification import DatasetCommandsClassification
    from dataset_detection import DatasetCommandsDetection
    from project_utils import get_project_info
    
    project_info = get_project_info()
    project_type = project_info["type"]

    if not dataset_name:
        confirmed_name = False
        
        while not confirmed_name:
            dataset_name = click.prompt(
                f"What is the dataset's name?"
            )
            
            if click.confirm(f'{Fore.CYAN}{dataset_name}{Fore.RESET}, is this name right?'):
                confirmed_name = True

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
cli.add_command(wrath)
cli.add_command(dataset)
cli.add_command(model)

def main():
    cli()

if __name__ == "__main__":
    main()