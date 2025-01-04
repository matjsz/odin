       
import logging

from colorama import Fore
from odin_vision.api.chronicle.controller import OdinChroniclesController
from odin_vision.api.dataset.controller import OdinDatasetsController
from odin_vision.api.internal.exceptions import CouldNotFindProjectError, ProjectTypeInvalidError
from odin_vision.api.model.controller import OdinModelsController
from odin_vision.api.project.base import OdinProjectBase
from odin_vision.api.project.classification import new_classification_project
from odin_vision.api.project.detection import new_detection_project

class OdinProject(OdinProjectBase):
    def __init__(
      self,
      name: str | None = None,
      type: str = "detection",
      allow_creation: bool = False,
      create_examples: bool = True,
      project_dir: str | None = None,
      verbose: bool = False,
      **kwargs  
    ):
        # Project general info loading and input checks
        self.verbose = verbose
        self.project_data = {}
        
        self._check_project_name(name)
        self._check_project_type(allow_creation, type)
        
        self.name = name
        self.type = type
        
        project_builders = {
            "classification": new_classification_project,
            "detection": new_detection_project
        }
        try:
            self._project_builder = project_builders[type]
        except:
            raise ProjectTypeInvalidError(type)
        
        self.allow_creation = allow_creation
        project_exists, project_path = self._check_project_exists(name)
        
        if project_dir:
            self.path = project_dir
        else:
            self.path = project_path
        
        # Checks if project exists, otherwise create a new one if allow_creation is True
        if not project_exists:
            if allow_creation:
                self._create_new_project(create_examples)
            else:
                raise CouldNotFindProjectError(project_path)
        else:
            self.project_data = self._retrieve_project_data()
            self.type = self.project_data["type"]
            
            self._project_builder = project_builders[self.type]
            
            if verbose:
                logging.info(f"Project {Fore.CYAN}{name}{Fore.RESET} succesfully loaded.")
        
        # VITAL INFO LOADING ========================================/
        
        # Loads datasets info
        self.datasets = OdinDatasetsController(self)
        
        # Loads chronicles info
        self.chronicles = OdinChroniclesController(self)
        
        # Loads weights info
        self.models = OdinModelsController(self)
    
    def see(self):
        print(f"\n{Fore.CYAN}{self.name}{Fore.RESET}\n")
        print(f"Projet Type: {Fore.CYAN}{self.type}{Fore.RESET}")
        print(f"Project Path: {Fore.CYAN}{self.path}{Fore.RESET}")
        
        if len(self.datasets.datasets) == 0:
            print(f"Datasets: {Fore.YELLOW}Empty{Fore.RESET}")
        else:
            dataset_names = list(map(lambda dataset: f"{Fore.CYAN}{dataset.name}{Fore.RESET}", self.datasets.datasets))
            print(f"Datasets: {', '.join(dataset_names)}")
        
        if len(self.chronicles.chronicles) == 0:
            print(f"Chronicles: {Fore.YELLOW}Empty{Fore.RESET}")
        else:
            pass
        
        if len(self.models.staging_models) == 0:
            print(f"Staging Models: {Fore.YELLOW}Empty{Fore.RESET}")
        else:
            pass
        
        if len(self.models.staging_models) == 0:
            print(f"Published Models: {Fore.YELLOW}Empty{Fore.RESET}")
        else:
            pass
        
        print("")