from typing import Annotated, Literal


class OdinDataset:
    def __init__(self, name: str | None = None, project=None, path: str | None = None, allow_creation: bool = False, **kwargs):
        self.name = name
        self.allow_creation = allow_creation
        self.project = project
        
        self.path = path
        
        self.staging = []
        self.train = []
        self.val = []
        
        self.version = ""
        self.status = ""
        
    def publish(self, update_type: Literal["major", "minor", "fix"]="major", train: Annotated[int, "Value between 0 and 100"]=70, val: Annotated[int, "Value between 0 and 100"]=30):
        return