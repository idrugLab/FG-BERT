# FG-BERT
## Functional-Group-BERT
semi-supervised learning for molecular property prediction.

![image](https://github.com/idrugLab/FG-BERT/blob/main/FG-BERT.png)

# Requried package: 

## Example of FG-BERT environment installation：
--conda create -n FG-BERT python==3.7.0

--pip install tensorflow==2.3.0

--pip install rdkit

--pip install numpy

--pip install pandas

--conda install -c openbabel openbabel

--pip install matplotlib

--pip install hyperopt

--pip install scikit-learn

## Other packages can be installed with the latest version

-- pretrain: contains the codes for masked FG prediction pre-training task.

-- dataset_scoffold_random: contain the code to building dataset for pre-traing and fine-tuning.

-- utils: contain the code to convert molecules to graphs and set up the FG list.

--Data contains the dataset of each downstream task and the hyperparameters selected by each task.

-- bert_weightsMedium_20 ：The weights obtained after 20 epochs of model pre-training can be directly passed into the BERT model for downstream tasks, or the user can retrain the model to obtain the weights.
Users should first unzip the data file and place it in the right place. Then pre-training the FG-BERT for 20 epoch. After that, the classification or the regression file is used to predict specific molecular property.

# FG_nums and FG_list
## Statistics on the number of functional groups in the pre-trained corpus.
![image](https://github.com/idrugLab/FG-BERT/blob/main/FG-nums.png)


## List of functional groups used for pre-training of the FG-BERT model.
![image](https://github.com/idrugLab/FG-BERT/blob/main/fg_list.png)

# Example of use FG-BERT
## Pre-training example：
* utils.py
* model.py
* dataset_scaffold_random.py
* pretrain.py
* data.txt
* Create a folder named: ‘medium3_weights’ in the current folder of the code to hold the pre-trained weights for each epoch.
* python pretrain.py


## Fine-tuning example：

### BBBP dataset:
* utils.py
* model.py
* dataset_scaffold_random.py
* Class_hyperopt.py
* BBBP.csv
* A folder named "medium3_weights _BBBP" is created in the current code folder, the name of this folder should correspond to the name of the path in the arch dictionary, and is used to store the pre-training weights, which can be downloaded directly from this repository "bert_weightsMedium_20.h5". Another new folder called "classification_weights" is used to hold the optimal weights from the fine-tuning process.
After the model is run, you will have the weights for the 10 seeds in this folder. args = {"dense_dropout":0, "learning_rate":0.0000826682 , "batch_size":32, "num_heads":8}，This dictionary parameter needs to be modified for each classification task, and the parameters can be obtained from the FG_BERT_Hyperparameters.xlsx file.
* python Class_hyperopt.py



### ESOL dataset:
* utils.py
* model.py
* dataset_scaffold_random.py
* Reg_hyperopt.py
* ESOL.csv
* Create a folder named "medium3_weights _ESOL" in the current folder of the code The name of the folder should correspond to the pathname of the arch dictionary, and it is used to place the weights obtained from the pre-training, which can be downloaded directly from this repository, named 'bert_weightsMedium_20.h5 '. Another new folder named 'regression_weights' is created to hold the optimal weights from the fine-tuning process. After the model is run, you will get the weights of the 3 seeds in that folder. args = {"dense_dropout":0.05, "learning_rate":0.0000636859, "batch_size":16, "num_heads":8}，This dictionary parameter needs to be modified for each classification task, and the parameters can be obtained from the FG_BERT_Hyperparameters.xlsx file.
* python Reg_hyperopt.py

### ADMET dataset and cell-based phenotypic screening datasets:
* utils.py
* model.py
* dataset_scaffold_random.py
* Reg_hyperopt.py
* XXX.csv
* Similar to BBBP, the list of labels should be changed to the corresponding label column names, and the dataset_scoffold_random.py file should be modified and commented, with the Scaffold split commented, and the random split uncommented. seed seeds changed to [1,2,3,4,5,6,7,8,9,10]. The result can be obtained.
* python Class_hyperopt.py










