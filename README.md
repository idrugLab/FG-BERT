# FG-BERT

Functional-Graph-BERT

semi-supervised learning for molecular property prediction

requried package: tensorflow==2.3.0,rdkit,numpy,pandas, openbabel, Other packages can be installed with the latest version

-- pretrain: contains the codes for masked FG prediction pre-training task.

-- dataset_scoffold_random: contain the code to building dataset for pre-traing and fine-tuning.

-- utils: contain the code to convert molecules to graphs and set up the FG list.

--Data contains the dataset of each downstream task and the hyperparameters selected by each task.

-- bert_weightsMedium_20 ：The weights obtained after 20 epochs of model pre-training can be directly passed into the BERT model for downstream tasks, or the user can retrain the model to obtain the weights.

Users should first unzip the data file and place it in the right place. Then pre-training the FG-BERT for 20 epoch. After that, the classification or the regression file is used to predict specific molecular property.


