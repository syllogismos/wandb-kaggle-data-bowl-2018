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

It also tracks what commit you are on when you ran your script. I think this is very useful, when multiple people are working on the same repository and trying out various changes, it gets harder to track what gives you the best results. I wish this is even more strict, as in wandb should force you to checkin all the files before you run an experiment. But later I realized everytime you do a run, it stores the diff in the meta data of that run in wandb folder.

# Image Segmentation using UNET.
I'm playing with the dataset from this year's [kaggle data bowl](https://www.kaggle.com/c/data-science-bowl-2018). It is an image segmentation competition.

Integrating wandb and tracking your runs is very easy on your existing code base.

To get started you just have to do
```
import wandb  # Import the library
wandb.init()  # don't forget to initialise wandb

wandb.config = {} # hyperparameters and other stuff
wandb.log() # log your results that will be in your wandb dashboard in real time
```

I was playing with this kaggle [kernal](https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277) that implemented UNET.

As the model is implemented in keras, tracking results became so easy, as wandb comes with a simple callback that you pass to the training code.

Instead of logging all the metrics, I simply did
```
from wandb.keras import WandbCallback


# and pass that callback while training, to track the loss and other metrics you pass to the fit function automatically.
results = model.fit(X_train,
                    Y_train,
                    validation_split=0.1,
                    batch_size=8,
                    epochs=20,
                    callbacks=[earlystopper,
                               checkpointer,
                               WandbCallback()
                              ]
                    )
```

And when you simply run your script, your metrics start popping up in your dashboard.


Using config, I wanted to track some basic hyperparameters.

```
wandb.config.val_split = 0.1
wandb.config.batch_size = 10
wandb.config.epochs = 20
wandb.config.patience = 5
wandb.config.verbose = 1
```

Wandb also provides a way to upload images as a simple way to pass any image related debugging information. You can track how the training algorithm is improving and things like that.

For this I created another callback I pass along with the `WandbCallback` that contained the image related metadata I wanted to pass.

It looks like below.

```
class log_images_per_step(Callback):
    def on_epoch_end(self, epoch, logs={}):
        x = X_train[train_id]
        y = Y_train[train_id].astype(np.uint8)*255
        y_pred = self.model.predict(x[np.newaxis,:])
        y_pred_t=(y_pred > 0.5).astype(np.uint8)*255
        wandb.log({"examples": [wandb.Image(x, caption="x"),
                                wandb.Image(y, caption="y"),
                                wandb.Image(y_pred_t, caption="pred")]}, commit=False)
```

What the above callback does is basically after every epoch, it saves a sample image, and its given segmentation mask, and the prediction of the trained model till then.

You can log the image by passing a numpy array to `wandb.Image()` along with a caption, so that you can see it in the dashboard. The wandb image documentation can be found [here](https://docs.wandb.com/docs/images.html).

One issue I had was the callback I implemented and the WandbCallback are tracking the results as if images are from one epoch, and metrics and other tracking are from a different epoch. To fix this I simply had introduce `commit = False` in the `wandb.log` statement when I'm logging the images in my callback. This made sure that my images and the metrics are from the same epoch.


I also tried variants of the UNET, where I introduced an additional encoder and decoder layers in once case, and removed an encoder and decoder in another case.

In a way, the whole network is another hyperparameter, I wonder how I can track these layer additions and other stuff clearly. Maybe a simple description. Wandb automatically tracks what commit you ran an experiment on, but still I wish there is a better way of tracking network related changes.

You can also save any additional run specific data and upload it to wandb you can simply save any file in the folder specific to your run. You can simply access using this `wandb.run.dir`

I save the best model of the training using this and it gets uploaded after experiment ends.
```
os.path.join(wandb.run.dir, 'model-dsbowl2018-1.h5')
```

Not just this, it also automatically tracks the system related metrics like cpu, memory and storage without you having to do anything. This is also very important debugging information. When you have several variants each running on a different headless machine, you can just see from dashboard if a given experiment failed because of memory issues, or if the system resources are being used efficiently.

# Results

When you run this script, all the debug statements that show up on stdout will be streamed to your experiment run page. This way you can run several variants in various headless machines and debug and monitor them all from a single place in your wandb dashboard.

It stores all the metadata inside the run specific folder within the wandb folder. It contains things like, your hyperparameter config in `config.yaml`, the files you saved using `wandb.run.dir`, your debugging statements to stdout in `output.log`, the things you logged using `wandb.log` in `wandb-history.jsonl`, the diff of uncommitted code in `diff.patch`, the images in `media` folder, the system metrics like cpu, memory etc in `wandb-events.jsonl` and other wandb specific debugging information.

When the experiment is done it prints a cute summary of the experiment on command line like below.


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

And on the dashboard, it plots various metrics you are tracking like below picture.

![metrics](https://i.imgur.com/dvxLUQn.png)

The dashboard automatically creates a view for you based on what metrics you are tracking, but you can also change what type of plots and metrics you want to view.


And using the image widget you can see the images you tracked. Here are the images, based on the code we wrote.

![Imgur](https://i.imgur.com/lsQMf6S.png)
![Imgur](https://i.imgur.com/AcW6ofj.png)
![Imgur](https://i.imgur.com/SAhSVWP.png)
![Imgur](https://i.imgur.com/AsaQlzK.png)

And the system related metrics in its own view like below.
![Imgur](https://i.imgur.com/0Li94tz.png)


In the projects main page it shows you the summary of the results of all the runs like below
![Imgur](https://i.imgur.com/hnbi6YV.png)

# Summary
This is a breif introduction to wandb, how to use it and its dashboard. I played with an image segmentation problem, and with few changes in code, I was able to track my results and hyperparameters.

If you want to run this script, you can register to wandb from [here](wandb.com)

And you can get the data to train from the kaggle data bowl [page](ttps://www.kaggle.com/c/data-science-bowl-2018). Download and unzip the `stage1_train` and `stage1_test` sets and modify the `TRAIN_PATH` and `TEST_PATH` in the code accordingly.