from odin_vision.api.typings.dataset import OdinDataset


class OdinModel:
    def __init__(self, name: str | None=None, chronicle: str | None=None, dataset: str | None=None, project=None, model_path: str | None=None, status: str | None=None, **kwargs):
        self.name = name
        self.type: str | None = None
        self.chronicle = chronicle
        self.dataset = dataset
        self.project = project
        self.path = model_path
        self.status = status
        
    def train(self, dataset: str | OdinDataset | None=None, epochs: int=10, device: str="cpu", base_model: str | None=None, chronicle: str | None=None, subset: int=100, **kwargs):
        return
    
    def test(self):
        return
    
    def publish(self):
        return