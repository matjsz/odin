
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
