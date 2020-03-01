# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

# Getting Started

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using this dataset of 102 flower categories, you can see a few examples below.

![Flower](./assets/Flowers.png)

The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

We'll lead you through each part which you'll implement in Python.
When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.

### Prerequisites

Thinks you have to install or installed on your working machine:

* Python 3.7
* Numpy (win-64 v1.15.4)
* Pandas (win-64 v0.23.4)
* Matplotlib (win-64 v3.0.2)
* Jupyter Notebook
* Torchvision (win-64 v0.2.1)
* PyTorch (win-64 v0.4.1)

### Environment:
* [Miniconda](https://conda.io/miniconda.html) or [Anaconda](https://www.anaconda.com/download/)


### Installing

Use the package manager [pip](https://pip.pypa.io/en/stable/) or
[miniconda](https://conda.io/miniconda.html) or [Anaconda](https://www.anaconda.com/download/) to install your packages.  
A step by step guide to install the all necessary components in Anaconda for a Windows-64 System:
```bash
conda install -c conda-forge numpy
conda install -c conda-forge pandas
conda install -c conda-forge matplotlib
pip install torchvision
conda install -c pytorch pytorch
```

## Jupyter Notebook
* `Image Classifier Project.ipynb`

This jupyter notebook describe the whole project from udacity, from the beginning to the end.

## Running the project

The whole project is located in the file `Image Classifier Project.py` and it's include the training and the prediction part.
Based on this implementation the jupyter notebook was created from this.

```----- running with params -----
data directory:  flowers
save directory:  ./
architecture:    vgg13
learning rate:   0.001
hidden units:    500
epochs:          3
gpu:             True
-------------------------------
cnn neural network ...
  load image data ... done
  
  done
initialized.
start deep-learning in -gpu- mode ... 
epoch: 1/3..  training loss: 4.739..  validation loss: 3.941..  validation accuracy: 0.185
epoch: 1/3..  training loss: 4.100..  validation loss: 3.462..  validation accuracy: 0.261
epoch: 1/3..  training loss: 3.778..  validation loss: 2.932..  validation accuracy: 0.357
epoch: 1/3..  training loss: 3.443..  validation loss: 2.537..  validation accuracy: 0.423
epoch: 1/3..  training loss: 3.237..  validation loss: 2.177..  validation accuracy: 0.481
epoch: 1/3..  training loss: 3.069..  validation loss: 1.971..  validation accuracy: 0.512
epoch: 1/3..  training loss: 2.817..  validation loss: 1.731..  validation accuracy: 0.553
epoch: 1/3..  training loss: 2.751..  validation loss: 1.686..  validation accuracy: 0.581
epoch: 1/3..  training loss: 2.782..  validation loss: 1.538..  validation accuracy: 0.618
epoch: 1/3..  training loss: 2.391..  validation loss: 1.446..  validation accuracy: 0.627
epoch: 2/3..  training loss: 1.764..  validation loss: 1.364..  validation accuracy: 0.641
epoch: 2/3..  training loss: 2.310..  validation loss: 1.248..  validation accuracy: 0.668
epoch: 2/3..  training loss: 2.308..  validation loss: 1.183..  validation accuracy: 0.685
epoch: 2/3..  training loss: 2.345..  validation loss: 1.149..  validation accuracy: 0.697
epoch: 2/3..  training loss: 2.188..  validation loss: 1.096..  validation accuracy: 0.721
epoch: 2/3..  training loss: 2.222..  validation loss: 1.100..  validation accuracy: 0.709
epoch: 2/3..  training loss: 2.135..  validation loss: 1.033..  validation accuracy: 0.714
epoch: 2/3..  training loss: 2.136..  validation loss: 1.091..  validation accuracy: 0.694
epoch: 2/3..  training loss: 2.117..  validation loss: 0.981..  validation accuracy: 0.739
epoch: 2/3..  training loss: 2.213..  validation loss: 0.968..  validation accuracy: 0.746
epoch: 3/3..  training loss: 1.041..  validation loss: 0.886..  validation accuracy: 0.772
epoch: 3/3..  training loss: 1.999..  validation loss: 0.949..  validation accuracy: 0.744
epoch: 3/3..  training loss: 2.053..  validation loss: 0.849..  validation accuracy: 0.764
epoch: 3/3..  training loss: 1.859..  validation loss: 0.850..  validation accuracy: 0.756
epoch: 3/3..  training loss: 2.004..  validation loss: 0.851..  validation accuracy: 0.775
epoch: 3/3..  training loss: 1.925..  validation loss: 0.850..  validation accuracy: 0.751
epoch: 3/3..  training loss: 2.082..  validation loss: 0.865..  validation accuracy: 0.752
epoch: 3/3..  training loss: 1.996..  validation loss: 0.817..  validation accuracy: 0.799
epoch: 3/3..  training loss: 1.966..  validation loss: 0.826..  validation accuracy: 0.781
epoch: 3/3..  training loss: 1.805..  validation loss: 0.755..  validation accuracy: 0.809
-- done --
duration:  00:15:18
calculate accuracy on test ... done.
accuracy of the network on the 10000 test images: 77 %
duration:  00:00:18
save model to:  ./checkpoint.pth ... done
----- running with params -----
image file:      ./flowers/test/10/image_07117.jpg
load file:       checkpoint.pth
top k:           5
category names:  cat_to_name.json
gpu:             True
-------------------------------
load model from:  checkpoint.pth
create model ... done
initialize model ... done

--- prediction ---
load image data ... done
get prediction ... done.
 1 with 0.967 is globe thistle
 2 with 0.014 is spear thistle
 3 with 0.007 is common dandelion
 4 with 0.006 is great masterwort
 5 with 0.002 is artichoke
------------------
load image data ... done

```
![globe thistle](./assets/download.png)
![chart](./assets/download-1.png)  

After this I will explain more in details the training and prediction steps.

### Train the model

To train the neural network (CNN), start the first part of file `Image Classifier Project.py` marked as `train`.

### Parameters of training

To change to input folder, the output size and some other parameters for the neural network, you can adapt these global constants inside the python file.
  
 ```
# ---- set parameters ---------------
param_data_directory = "flowers"            # default: flowers
param_output_size = 102                     # 102 - original # 10 - test
param_save_file_name = "checkpoint.pth"     # checkpoint.pth
param_save_directory = "./"                 # ./
param_architecture = "vgg13"                # densenet121 or vgg13 or resnet18
param_learning_rate = 0.001                 # 0.001
param_hidden_units = 500                    # 500
param_epochs = 3                            # 3
param_gpu = True                            # True or False
# -----------------------------------
``` 
### Output of training

```
--Training starting--
Epoch 1/3.. Loss: 5.527.. Validation Loss: 4.772.. Accuracy: 0.120
Epoch 1/3.. Loss: 4.658.. Validation Loss: 4.018.. Accuracy: 0.199
Epoch 1/3.. Loss: 4.202.. Validation Loss: 3.596.. Accuracy: 0.272
Epoch 1/3.. Loss: 3.771.. Validation Loss: 3.393.. Accuracy: 0.322
Epoch 1/3.. Loss: 3.700.. Validation Loss: 3.202.. Accuracy: 0.336
Epoch 1/3.. Loss: 3.414.. Validation Loss: 3.012.. Accuracy: 0.359
Epoch 1/3.. Loss: 3.058.. Validation Loss: 2.833.. Accuracy: 0.384
Epoch 1/3.. Loss: 3.228.. Validation Loss: 2.589.. Accuracy: 0.455
Epoch 1/3.. Loss: 2.580.. Validation Loss: 2.454.. Accuracy: 0.450
Epoch 1/3.. Loss: 2.824.. Validation Loss: 2.174.. Accuracy: 0.511
Epoch 1/3.. Loss: 2.362.. Validation Loss: 2.053.. Accuracy: 0.525
Epoch 1/3.. Loss: 2.668.. Validation Loss: 1.975.. Accuracy: 0.549
Epoch 1/3.. Loss: 2.087.. Validation Loss: 1.898.. Accuracy: 0.553
Epoch 1/3.. Loss: 2.416.. Validation Loss: 1.799.. Accuracy: 0.560
Epoch 1/3.. Loss: 2.259.. Validation Loss: 1.726.. Accuracy: 0.600
Epoch 1/3.. Loss: 2.273.. Validation Loss: 1.617.. Accuracy: 0.629
Epoch 1/3.. Loss: 2.134.. Validation Loss: 1.554.. Accuracy: 0.633
Epoch 1/3.. Loss: 2.456.. Validation Loss: 1.502.. Accuracy: 0.658
Epoch 1/3.. Loss: 1.881.. Validation Loss: 1.402.. Accuracy: 0.682
Epoch 1/3.. Loss: 2.208.. Validation Loss: 1.338.. Accuracy: 0.672
Epoch 1/3.. Loss: 1.900.. Validation Loss: 1.307.. Accuracy: 0.680
Epoch 1/3.. Loss: 1.948.. Validation Loss: 1.295.. Accuracy: 0.690
Epoch 1/3.. Loss: 1.733.. Validation Loss: 1.292.. Accuracy: 0.698
Epoch 1/3.. Loss: 1.842.. Validation Loss: 1.223.. Accuracy: 0.712
Epoch 1/3.. Loss: 1.748.. Validation Loss: 1.134.. Accuracy: 0.728
Epoch 1/3.. Loss: 1.900.. Validation Loss: 1.079.. Accuracy: 0.744
Epoch 1/3.. Loss: 1.789.. Validation Loss: 1.071.. Accuracy: 0.746
Epoch 1/3.. Loss: 1.577.. Validation Loss: 1.116.. Accuracy: 0.734
Epoch 1/3.. Loss: 2.023.. Validation Loss: 1.112.. Accuracy: 0.709
Epoch 1/3.. Loss: 2.031.. Validation Loss: 1.031.. Accuracy: 0.768
Epoch 1/3.. Loss: 1.458.. Validation Loss: 1.009.. Accuracy: 0.756
Epoch 1/3.. Loss: 1.646.. Validation Loss: 0.945.. Accuracy: 0.778
Epoch 1/3.. Loss: 1.894.. Validation Loss: 0.962.. Accuracy: 0.765
Epoch 1/3.. Loss: 1.473.. Validation Loss: 0.928.. Accuracy: 0.770
Epoch 1/3.. Loss: 1.368.. Validation Loss: 0.875.. Accuracy: 0.783
Epoch 1/3.. Loss: 1.699.. Validation Loss: 0.857.. Accuracy: 0.778
Epoch 1/3.. Loss: 1.415.. Validation Loss: 0.855.. Accuracy: 0.777
Epoch 1/3.. Loss: 1.536.. Validation Loss: 0.816.. Accuracy: 0.795
Epoch 1/3.. Loss: 1.354.. Validation Loss: 0.820.. Accuracy: 0.804
Epoch 1/3.. Loss: 1.289.. Validation Loss: 0.833.. Accuracy: 0.784
Epoch 1/3.. Loss: 1.424.. Validation Loss: 0.793.. Accuracy: 0.794
Epoch 2/3.. Loss: 0.916.. Validation Loss: 0.768.. Accuracy: 0.793
Epoch 2/3.. Loss: 1.048.. Validation Loss: 0.753.. Accuracy: 0.813
Epoch 2/3.. Loss: 0.934.. Validation Loss: 0.738.. Accuracy: 0.811
Epoch 2/3.. Loss: 1.072.. Validation Loss: 0.710.. Accuracy: 0.802
Epoch 2/3.. Loss: 0.937.. Validation Loss: 0.726.. Accuracy: 0.797
Epoch 2/3.. Loss: 0.955.. Validation Loss: 0.745.. Accuracy: 0.802
Epoch 2/3.. Loss: 1.052.. Validation Loss: 0.728.. Accuracy: 0.808
Epoch 2/3.. Loss: 0.877.. Validation Loss: 0.693.. Accuracy: 0.823
Epoch 2/3.. Loss: 0.868.. Validation Loss: 0.699.. Accuracy: 0.809
Epoch 2/3.. Loss: 1.138.. Validation Loss: 0.689.. Accuracy: 0.827
Epoch 2/3.. Loss: 0.826.. Validation Loss: 0.707.. Accuracy: 0.805
Epoch 2/3.. Loss: 1.032.. Validation Loss: 0.742.. Accuracy: 0.795
Epoch 2/3.. Loss: 0.922.. Validation Loss: 0.706.. Accuracy: 0.812
Epoch 2/3.. Loss: 1.097.. Validation Loss: 0.668.. Accuracy: 0.826
Epoch 2/3.. Loss: 0.869.. Validation Loss: 0.734.. Accuracy: 0.815
Epoch 2/3.. Loss: 1.137.. Validation Loss: 0.705.. Accuracy: 0.827
Epoch 2/3.. Loss: 1.143.. Validation Loss: 0.653.. Accuracy: 0.827
Epoch 2/3.. Loss: 0.760.. Validation Loss: 0.638.. Accuracy: 0.841
Epoch 2/3.. Loss: 1.063.. Validation Loss: 0.656.. Accuracy: 0.836
Epoch 2/3.. Loss: 0.970.. Validation Loss: 0.637.. Accuracy: 0.843
Epoch 2/3.. Loss: 0.922.. Validation Loss: 0.639.. Accuracy: 0.849
Epoch 2/3.. Loss: 0.931.. Validation Loss: 0.616.. Accuracy: 0.843
Epoch 2/3.. Loss: 0.775.. Validation Loss: 0.611.. Accuracy: 0.836
Epoch 2/3.. Loss: 0.887.. Validation Loss: 0.617.. Accuracy: 0.845
Epoch 2/3.. Loss: 0.793.. Validation Loss: 0.622.. Accuracy: 0.823
Epoch 2/3.. Loss: 1.044.. Validation Loss: 0.631.. Accuracy: 0.828
Epoch 2/3.. Loss: 1.072.. Validation Loss: 0.649.. Accuracy: 0.818
Epoch 2/3.. Loss: 0.752.. Validation Loss: 0.637.. Accuracy: 0.824
Epoch 2/3.. Loss: 0.857.. Validation Loss: 0.641.. Accuracy: 0.827
Epoch 2/3.. Loss: 0.965.. Validation Loss: 0.609.. Accuracy: 0.831
Epoch 2/3.. Loss: 0.942.. Validation Loss: 0.582.. Accuracy: 0.846
Epoch 2/3.. Loss: 0.802.. Validation Loss: 0.592.. Accuracy: 0.848
Epoch 2/3.. Loss: 0.746.. Validation Loss: 0.589.. Accuracy: 0.841
Epoch 2/3.. Loss: 1.054.. Validation Loss: 0.611.. Accuracy: 0.835
Epoch 2/3.. Loss: 0.879.. Validation Loss: 0.610.. Accuracy: 0.843
Epoch 2/3.. Loss: 0.787.. Validation Loss: 0.612.. Accuracy: 0.835
Epoch 2/3.. Loss: 0.788.. Validation Loss: 0.568.. Accuracy: 0.858
Epoch 2/3.. Loss: 0.752.. Validation Loss: 0.561.. Accuracy: 0.848
Epoch 2/3.. Loss: 0.742.. Validation Loss: 0.529.. Accuracy: 0.856
Epoch 2/3.. Loss: 0.761.. Validation Loss: 0.537.. Accuracy: 0.846
Epoch 2/3.. Loss: 0.877.. Validation Loss: 0.536.. Accuracy: 0.843
Epoch 3/3.. Loss: 0.677.. Validation Loss: 0.544.. Accuracy: 0.857
Epoch 3/3.. Loss: 0.672.. Validation Loss: 0.581.. Accuracy: 0.844
Epoch 3/3.. Loss: 0.670.. Validation Loss: 0.560.. Accuracy: 0.844
Epoch 3/3.. Loss: 0.593.. Validation Loss: 0.557.. Accuracy: 0.849
Epoch 3/3.. Loss: 0.549.. Validation Loss: 0.539.. Accuracy: 0.854
Epoch 3/3.. Loss: 0.647.. Validation Loss: 0.500.. Accuracy: 0.862
Epoch 3/3.. Loss: 0.672.. Validation Loss: 0.504.. Accuracy: 0.866
Epoch 3/3.. Loss: 0.607.. Validation Loss: 0.489.. Accuracy: 0.869
Epoch 3/3.. Loss: 0.623.. Validation Loss: 0.489.. Accuracy: 0.867
Epoch 3/3.. Loss: 0.657.. Validation Loss: 0.497.. Accuracy: 0.865
Epoch 3/3.. Loss: 0.719.. Validation Loss: 0.511.. Accuracy: 0.857
Epoch 3/3.. Loss: 0.776.. Validation Loss: 0.537.. Accuracy: 0.847
Epoch 3/3.. Loss: 0.465.. Validation Loss: 0.537.. Accuracy: 0.859
Epoch 3/3.. Loss: 0.567.. Validation Loss: 0.524.. Accuracy: 0.861
Epoch 3/3.. Loss: 0.580.. Validation Loss: 0.536.. Accuracy: 0.851
Epoch 3/3.. Loss: 0.526.. Validation Loss: 0.545.. Accuracy: 0.854
Epoch 3/3.. Loss: 0.718.. Validation Loss: 0.512.. Accuracy: 0.867
Epoch 3/3.. Loss: 0.594.. Validation Loss: 0.515.. Accuracy: 0.865
Epoch 3/3.. Loss: 0.583.. Validation Loss: 0.504.. Accuracy: 0.863
Epoch 3/3.. Loss: 0.719.. Validation Loss: 0.494.. Accuracy: 0.866
Epoch 3/3.. Loss: 0.467.. Validation Loss: 0.531.. Accuracy: 0.861
Epoch 3/3.. Loss: 0.785.. Validation Loss: 0.493.. Accuracy: 0.862
Epoch 3/3.. Loss: 0.602.. Validation Loss: 0.490.. Accuracy: 0.866
Epoch 3/3.. Loss: 0.796.. Validation Loss: 0.509.. Accuracy: 0.868
Epoch 3/3.. Loss: 0.851.. Validation Loss: 0.512.. Accuracy: 0.869
Epoch 3/3.. Loss: 0.574.. Validation Loss: 0.564.. Accuracy: 0.860
Epoch 3/3.. Loss: 0.709.. Validation Loss: 0.582.. Accuracy: 0.850
Epoch 3/3.. Loss: 0.733.. Validation Loss: 0.601.. Accuracy: 0.843
Epoch 3/3.. Loss: 0.667.. Validation Loss: 0.570.. Accuracy: 0.862
Epoch 3/3.. Loss: 0.666.. Validation Loss: 0.535.. Accuracy: 0.873
Epoch 3/3.. Loss: 0.808.. Validation Loss: 0.545.. Accuracy: 0.872
Epoch 3/3.. Loss: 0.657.. Validation Loss: 0.540.. Accuracy: 0.861
Epoch 3/3.. Loss: 0.594.. Validation Loss: 0.537.. Accuracy: 0.851
Epoch 3/3.. Loss: 0.780.. Validation Loss: 0.552.. Accuracy: 0.849
Epoch 3/3.. Loss: 0.641.. Validation Loss: 0.538.. Accuracy: 0.862
Epoch 3/3.. Loss: 0.683.. Validation Loss: 0.528.. Accuracy: 0.861
Epoch 3/3.. Loss: 0.724.. Validation Loss: 0.512.. Accuracy: 0.871
Epoch 3/3.. Loss: 0.662.. Validation Loss: 0.500.. Accuracy: 0.877
Epoch 3/3.. Loss: 0.675.. Validation Loss: 0.554.. Accuracy: 0.851
Epoch 3/3.. Loss: 0.831.. Validation Loss: 0.548.. Accuracy: 0.856
Epoch 3/3.. Loss: 0.845.. Validation Loss: 0.625.. Accuracy: 0.848
Saved checkpoint!
```
