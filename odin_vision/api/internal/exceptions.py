from colorama import Fore
from odin_vision.constants import PROJECT_TYPES

class ProjectNameInvalidError(Exception):
    """Exception raised for invalid project name.

    Attributes:
        name - the name that caused the error
    """

    def __init__(self, name):
        self.name = name
        self.message = f"The folowing name wasn't accepted as a valid {Fore.CYAN}Odin{Fore.RESET} project name: {Fore.LIGHTRED_EX}{name}{Fore.RESET}"
        super().__init__(self.message)
        
class ProjectTypeInvalidError(Exception):
    """Exception raised for invalid project type.

    Attributes:
        type - the type that caused the error
    """

    def __init__(self, type):
        self.type = type
        
        project_pretty_types = list(map(lambda valid_type: f"{Fore.CYAN}{valid_type}{Fore.RESET}", PROJECT_TYPES))
        
        self.message = f"The folowing type wasn't accepted as a valid {Fore.CYAN}Odin{Fore.RESET} project type: {Fore.LIGHTRED_EX}{type}{Fore.RESET}. It may be one of {', '.join(project_pretty_types)}"
        super().__init__(self.message)
        
class CouldNotCreateProjectInPathError(Exception):
    """Exception raised for invalid project name.

    Attributes:
        path - the path that caused the error
    """

    def __init__(self, path):
        self.path = path
        self.message = f"Couldn't create a new project at {Fore.LIGHTRED_EX}{path}{Fore.RESET}. Check the path again or open a new issue at the official repository: https://github.com/matjsz/odin."
        super().__init__(self.message)
        
class CouldNotCreateDatasetInPathError(Exception):
    """Exception raised for invalid project name.

    Attributes:
        path - the path that caused the error
    """

    def __init__(self, path):
        self.path = path
        self.message = f"Couldn't create a new dataset at {Fore.LIGHTRED_EX}{path}{Fore.RESET}. Check the path again or open a new issue at the official repository: https://github.com/matjsz/odin."
        super().__init__(self.message)        
        
class CouldNotFindProjectError(Exception):
    """Exception raised when project is not found and allow_creation is not True.

    Attributes:
        name - the project name that caused the error
    """

    def __init__(self, path):
        self.path = path
        self.message = f"Couldn't find a project at {Fore.LIGHTRED_EX}{path}{Fore.RESET}. Check the path again or set {Fore.CYAN}allow_creation{Fore.RESET} to true at your {Fore.CYAN}OdinProject{Fore.RESET}."
        super().__init__(self.message)
        
class CouldNotFindDatasetError(Exception):
    """Exception raised when dataset is not found and allow_creation is not True.

    Attributes:
        path - the dataset path that caused the error
    """

    def __init__(self, path):
        self.path = path
        self.message = f"Couldn't find a dataset at {Fore.LIGHTRED_EX}{path}{Fore.RESET}. Check the path again or set {Fore.CYAN}allow_creation{Fore.RESET} to True at your {Fore.CYAN}OdinDataet{Fore.RESET} instance."
        super().__init__(self.message)
        
class CouldNotFindDatasetByNameError(Exception):
    """Exception raised when dataset is not found by name and allow_creation is not True.

    Attributes:
        name - the project name that caused the error
    """

    def __init__(self, name):
        self.name = name
        self.message = f"Couldn't find a dataset named as {Fore.LIGHTRED_EX}{name}{Fore.RESET}. Check the name again or set {Fore.CYAN}allow_creation{Fore.RESET} to True at your {Fore.CYAN}OdinProject.datasets.get{Fore.RESET} function."
        super().__init__(self.message)
        
class CouldNotFindChronicleByNameError(Exception):
    """Exception raised when chronicle is not found by name and allow_creation is not True.

    Attributes:
        name - the project name that caused the error
    """

    def __init__(self, name):
        self.name = name
        self.message = f"Couldn't find a chronicle named as {Fore.LIGHTRED_EX}{name}{Fore.RESET}. Check the name again."
        super().__init__(self.message)
        
class DatasetNotInformedError(Exception):
    """Exception raised when dataset is not informed when performing a dataset action.
    """

    def __init__(self):
        self.message = f"A dataset wasn't informed to the function. Please provide a {Fore.CYAN}dataset name{Fore.RESET} or a {Fore.CYAN}dataset instance{Fore.RESET}."
        super().__init__(self.message)
        
class InvalidSplitPercentagesError(Exception):
    """Exception raised when dataset split percentages cannot sum to 100.

    Attributes:
        train - the train split percentage
        val - the val split percentage
    """

    def __init__(self, train, val):
        self.train = train
        self.val = val
        self.message = f"The split values ({Fore.LIGHTRED_EX}train{Fore.RESET}: {train} | {Fore.LIGHTRED_EX}val{Fore.RESET}: {val}) doesn't sum to 100, it is either below or above 100."
        super().__init__(self.message)
        
class ModelNotInformedError(Exception):
    """Exception raised when model is not informed when performing a model action.
    """

    def __init__(self):
        self.message = f"A model wasn't informed to the function. Please provide a {Fore.CYAN}model name{Fore.RESET} or a {Fore.CYAN}model instance{Fore.RESET}."
        super().__init__(self.message)