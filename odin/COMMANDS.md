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

## Naive version

```
$ odin train [dataset] --epochs=30 --chronicle={chronicle} --subset=50 (percentage)

Training with {subset}% of the dataset...
Defined as "naive" version.
Trained to chronicle {chronicle}

You can test this version by using the command odin test {chronicle}
```

```
$ odin test [chronicle]
```

If all went okay, train the wise version.

## Wise version

```
$ odin train [dataset] --epochs=30 --chronicle={chronicle}

Training with 100% of the dataset...
Defined as "wise"
Trained to chronicle {chronicle}

You can test this version by using the command odin test {chronicle}
```

It will generate the weight `{year}_{month}_{day}_{H}{M}{S}.pt` at `chronicles/{chronicle}/weights`.

```
$ odin test [chronicle]
```

Time to publish it!

# Publishing a new model

```
$ odin publish [model]

Publishing model [model]...
Model [model] published succesfully.
Generated version '{project}_{year}_{month}_{day}_{chronicle}.pt'
```

Example model name: `car_detector_2024-12-1_epic-ninja.pt`