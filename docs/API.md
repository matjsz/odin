## Starting a project

```python
from odin_vision.api.pipeline import OdinProject

my_project = OdinProject(name="my-project", type="classification").create()
```

```python
my_project = OdinProject(name="my-project", type="classification", allow_creation=True)
```

## Retrieving an existing project

```python
my_project = OdinProject(name="my-project")
```

## Creating a dataset

```python
my_project.datasets.create("dataset1") # Returns OdinDataset
my_dataset: OdinDataset = my_project.datasets.get(by_name="dataset1", allow_creation=True)
```

## Publishing a dataset

```python
my_dataset.publish()
```

```python
my_project.datasets.publish(dataset="dataset1")
my_project.datasets.publish(dataset=my_dataset)
```

## Training a new model

```python
# Returns a OdinModel
my_project.models.train(dataset="dataset1")
```

```python
my_project.models.train(dataset=my_dataset)
```

## Getting a trained model
```python
my_project.models.get("all", "staging", by_dataset="dataset1") # Returns list[OdinModel]
my_project.models.get("all", "staging", by_dataset=my_dataset) # Returns list[OdinModel]
my_project.models.get("last", "published", by_dataset="dataset1") # Returns OdinModel
my_project.models.get("last", "published", by_dataset=my_dataset) # Returns OdinModel

my_project.models.get("all", "staging", by_chronicle="epic-hero") # Returns list[OdinModel]
my_project.models.get("all", "staging", by_chronicle=OdinChronicle) # Returns list[OdinModel]
my_project.models.get("last", "published", by_chronicle="epic-hero") # Returns OdinModel
my_project.models.get("last", "published", by_chronicle=OdinChronicle) # Returns OdinModel

my_model: OdinModel = my_project.models.get(by_name="model1")
```

## Testing a trained model
```python
results: OdinTestReport = my_model.test()
```

```python
results: OdinTestReport = my_project.models.test(model="dataset1")
results: OdinTestReport = my_project.models.test(model=my_model)
```

## Publishing a trained model

```python
my_model = my_project.models.get(by_name="model1")
my_model.publish()
```

```python
my_project.models.publish(model="model1")
my_project.models.publish(model=my_model)
```