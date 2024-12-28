# Starting a new project

```
$ odin start [project]
```

Starts a new project with a whole structure to follow.

# Creating a new dataset

```
$ odin dataset create
```

# Training a new model

```
$ odin train [dataset] --epochs=30 --chronicle={chronicle}

Training...
Trained to chronicle {chronicle}
Consider publishing the weights to a final version
``` 

```
$ odin test [chronicle]
```