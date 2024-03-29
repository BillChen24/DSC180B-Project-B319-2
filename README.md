# DSC180B-Project-B319-2
# Domain Adaptation of CNN in Animal Classification Task

### For Test Trails:
After downloading the githubt repo (including the sample data in data folder), run the following code in terminal
```
python run.py test
```
This code will train a custom CNN model on the sample data and generate a loss curve over epochs.

The loss plot will be saved in the path printed at the end of the execution.

The trained model will be saved in the path printed at the end of the execution.

Code will create a "result/" folder if such folder doesn't exist in the local repository and store the loss plot and model in this folder

#### After Runing
To clear all output files, run the following command in terminal:
```
rm -r result/
```


### To Get Whole dataset
Create environment for loading iwildcam dataset 

0: Open a terminal 

1: ssh in tothe dsmlp environment:
ssh <user_name>@dsmlp-login.ucsd.edu

Example: ssh zhc023@dsmlp-login.ucsd.edu

2: launch-scipy-ml.sh -c 8 -m 50 -i billchen24/dsc180b-project -P Always // request 8 cpu and 50 GB, and create specific environment with my docker image 

3: Create new terminal 

4: ssh -N -L 8889:127.0.0.1:16585 zhc023@dsmlp-login.ucsd.edu

5: Go to http://localhost:8889/user/zhc023/tree/ 

### To Run the Full Experiment
Run the following command
```
python run.py main 10
```
"10" specified the maximum number of epoch the model will train, Feel free to update it to any value.
Some other hyperparameters can be view and change in "run.py"

Similar to the test trail, all output from can be removed by 
```
rm -r result/
```
