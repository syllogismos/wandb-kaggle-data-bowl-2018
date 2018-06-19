# Introduction to Weights & Biases


<b>wandb</b> is a tool that helps track and visualize machine learning model training at scale.

The best thing about wandb is that it fits into your current workflow with just a few lines of code and it provides an enormous value. Often times you run the same model with hundred different settings, but in the end you don't have a proper way of tracking what settings worked and what didn't. Wandb aims to solve this problem without you having to do so much.

Here are few benifits according to the documentation.

1. Tracking, saving and reproducing models.
2. Visualizing results across models.
3. Debugging and visualizing hardware performance issues.
4. Automating large-scale hyperparameter search.


# Installation.

You can simply install wandb using pip, My environment is python3
```
pip install wandb
```

The only problem I had with the installation of the package is with gcc.

On OSX you will see something like
```
error: command 'gcc' failed with exit status 1 - Google Search
```

You can fix it with this
```
xcode-select --install
```

On Ubuntu you will see something like 
```
Unable to execute gcc: No such file or directory
```
And you can fix it by installing gcc
```
sudo apt-get install gcc
```

# Signup to wandb and setting up your machine.
You can get your free wandb account by signing up here https://app.wandb.ai/

And run `wandb login` from your command line to setup wandb in your machine. On normal machine it automatically authorises your machine by opening your browser and making you login, on a headless machine it will ask you to enter your API key that you can find in your profile.


# Setting up your Experiment

You can initialise wandb in the experiment directory by doing this
```
wandb init
```
It asks you to create the name of the experiment in the wandb dashboard. All the wandb metadata, and the experiment data you log through wandb commands will be stored in this directory. And after the experiment is run, it will be uploaded to your dashboard.

It also tracks what commit you are on when you ran your script. I think this is very useful, when multiple people are working on the same repository and trying out various changes, it gets harder to track what gives you the best results. I wish this is even more strict, as in wandb should force you to checkin all the files before you run an experiment.



## Logs
```
wandb: Waiting for wandb process to finish, PID 62741
wandb: Program ended.
wandb: Run summary:
wandb:           loss 0.07844819274930219
wandb:   val_mean_iou 0.6689269658344895
wandb:       mean_iou 0.6655849722487417
wandb:       val_loss 0.07152052184761461
wandb: Run history:
wandb:       mean_iou ▁▁▁▁▁▁▁▁▁▁▁▂▂▃▃▄▄▅▅▆▆▆▇▇▇▇▇███
wandb:       val_loss █▅▆▅▅▅▄▄▃▃▃▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:          epoch ▁▁▁▂▂▂▂▃▃▃▃▄▄▄▄▅▅▅▅▆▆▆▆▇▇▇▇███
wandb:           loss █▅▄▄▄▄▄▃▃▃▃▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:   val_mean_iou ▁▁▁▁▁▁▁▁▁▁▁▂▂▃▃▄▄▅▅▆▆▆▇▇▇▇▇███
wandb: Waiting for final file modifications.
wandb: Syncing files in wandb/run-20180619_134014-4kcu8duf:
wandb:   wandb-debug.log
wandb:   model-best.h5
wandb:   wandb-metadata.json
wandb:   model-dsbowl2018-1.h5
wandb:
wandb: Verifying uploaded files... verified!
wandb: Synced https://app.wandb.ai/syllogismos/kaggle-data-science-bowl-2018/runs/4kcu8duf
```

```
wandb: Waiting for wandb process to finish, PID 84625
wandb: Program ended.
wandb: Run summary:
wandb:       mean_iou 0.7422491321911662
wandb:           loss 0.08380086232551295
wandb:   val_mean_iou 0.7450126258294973
wandb:       val_loss 0.0756254298473472
wandb: Run history:
wandb:   val_mean_iou ▁▁▁▂▃▄▅▅▆▆▆▇▇▇▇█████
wandb:           loss █▅▄▃▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁
wandb:       mean_iou ▁▁▁▂▃▄▅▅▆▆▆▇▇▇▇█████
wandb:       val_loss █▆▄▂▂▂▂▁▂▁▁▁▁▁▁▁▁▁▁▁
wandb:          epoch ▁▁▂▂▂▃▃▄▄▄▅▅▅▆▆▇▇▇██
wandb: Waiting for final file modifications.
wandb: Syncing files in wandb/run-20180619_154604-eyegic7r:
wandb:   wandb-metadata.json
wandb:   wandb-debug.log
wandb:   model-dsbowl2018-1.h5
wandb:   config.yaml
wandb:   model-best.h5
wandb:
wandb: Verifying uploaded files... verified!
wandb: Synced https://app.wandb.ai/syllogismos/kaggle-data-science-bowl-2018/runs/eyegic7r
```

```
wandb: Waiting for wandb process to finish, PID 11681
wandb: Program ended.
wandb: Run summary:
wandb:       val_loss 0.07321353884997653
wandb:   val_mean_iou 0.7602148456359977
wandb:       mean_iou 0.7578663706384092
wandb:           loss 0.0780772963788972
wandb: Run history:
wandb:       mean_iou ▁▁▁▂▄▄▅▆▆▆▇▇▇▇▇█████
wandb:       val_loss █▅▃▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁
wandb:   val_mean_iou ▁▁▂▃▄▅▅▆▆▆▇▇▇▇▇█████
wandb:          epoch ▁▁▂▂▂▃▃▄▄▄▅▅▅▆▆▇▇▇██
wandb:           loss █▅▃▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb: Waiting for final file modifications.
wandb: Syncing files in wandb/run-20180619_161431-ym4us3xm:
wandb:   model-best.h5
wandb:   wandb-metadata.json
wandb:   config.yaml
wandb:   model-dsbowl2018-1.h5
wandb:   wandb-debug.log
wandb:
wandb: Verifying uploaded files... verified!
wandb: Synced https://app.wandb.ai/syllogismos/kaggle-data-science-bowl-2018/runs/ym4us3xm
```
