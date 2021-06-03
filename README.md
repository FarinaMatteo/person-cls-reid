# person-cls-reid
Project for the Deep Learning course at University of Trento, MSc in Artificial Intelligence / Computer Science.
The goal of the project is to perform both *Classification* and *Re-Identification* tasks on the famous Market-1501 Dataset.


## Introduction  
This **README** contains instructions on what is the main content of this repo and how to get your system up and running to execute the code.
  
Furthermore, it contains every needed file for the final delivery:  
- *classification_test.csv*: containing the output of the Attribute Recognition Pipeline on *dataset/test*;  
- *reid_test.txt*: containing the output of the Re-Identification pipeline for each *query* in *dataset/queries*, *dataset/test* being our *Gallery*;  

## Software Requirements and Installation  
This repository has been tested under the Python 3.6 interpreter, while packages have been installed through `pip` inside a
`Conda` environment. Below you can find two alternative installation methods:  

- #### Anaconda Environment  
    ```$ conda env create -f conda_environment.yml```
- #### Manual Pip Installation  
    ```$ pip install -r requirements.txt```

## Classification Task
In order to train a classifier, launch the following command:  
```(your-env) $ python train_classifier.py```  
This will generate a new `.pth` file inside *models* with the current experiment name (you can configure it in the last line of `train_classifier.py`, 
in the `exp_name` argument of the `main` function.)

In order to launch the attribute recognition pipeline using a pre-trained architecture (*)(by default *networks/model.pth*), run the following:  
```(your-env) $ python attribute_recognition.py```  
This will generate a *classification_test.csv* file containing predictions for each image in *dataset/test*.

(*) In order to execute the attribute recognition pipeline, you can use [these pretrained weights](https://drive.google.com/file/d/1Iyw54v5mWTEF5eM2TLuBTmKUnv0FZ4P3/view?usp=sharing).
Please download them and place them as *networks/model.pth* (only available with a UniTN Google Account).

## Re-Identification Task
Since no additional training procedure is involved, you can directly execute the Re-Identification pipeline provided that you have 
a suitable classifier.  

To run the Re-Id Evaluation, execute:  
```(your-env) $ python reid_validation.py```
This will output your Re-ID Accuracy as well as your mAP.  

To run the Re-Id on *dataset/queries*, execute:  
```(your-env) $ python reid_queries.py```
This will produce a *reid_test.txt* file containing the Re-Identification results for each image in *dataset/queries*.  

## Contributions  
The authors of this repository are:  
- [Cunegatti Elia](https://github.com/eliacunegatti)  
- [Diprima Federico](https://github.com/fedediprima)
- Farina Matteo (me)


