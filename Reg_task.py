import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import os
import pandas as pd
import numpy as np
from tensorflow.keras.constraints import max_norm
from dataset_scoffold_random import Graph_Regression_Dataset
from sklearn.metrics import roc_auc_score
from model import  PredictModel,BertModel
from hyperopt import fmin, tpe, hp

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_DETERMINISTIC_OPS'] = '1'
keras.backend.clear_session()

def main(seed,args):
    # tasks = ['ESOL', 'FreeSolv', 'Lipophilicity', 'Malaria', 'cep']
    
    task = 'ESOL'
    print(task)

    if task == 'ESOL':
        label = ['measured log solubility in mols per litre']
    
    elif task == 'FreeSolv':
        label = ['expt']
    
    elif task == 'Lipophilicity':
        label = ['exp']

    elif task == 'Malaria':
        label = ['PCE']

    elif task == 'cep':
        label = ['activity']


    vocab_size = 18
    trained_epoch = 10
    num_layers = 6
    d_model = 256
    addH = True
    dff = d_model * 2
    seed = seed
    arch  = {'name': 'Medium', 'path': 'medium3_weights_ESOL'}

    dense_dropout = args['dense_dropout']
    learning_rate = args['learning_rate']
    batch_size = args['batch_size']
    num_heads = args['num_heads']

    pretraining = True
    pretraining_str = 'pretraining' if pretraining else ''

    np.random.seed(seed=seed)
    tf.random.set_seed(seed=seed)
    graph_dataset = Graph_Regression_Dataset('ESOL.csv', smiles_field='smiles',
                                                           label_field=label,normalize=True,seed=seed,batch_size=batch_size,a=len(label),max_len=500,addH=True)
        
    train_dataset, test_dataset,val_dataset = graph_dataset.get_data()
    

    x, adjoin_matrix, y = next(iter(train_dataset.take(1)))
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    model = PredictModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,a=len(label),
                        dense_dropout = dense_dropout)
    if pretraining:
        temp = BertModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size)
        pred = temp(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)
        temp.load_weights(arch['path']+'/bert_weights{}_{}.h5'.format(arch['name'],trained_epoch))
        temp.encoder.save_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
        del temp

        pred = model(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)

        model.encoder.load_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
        print('load_wieghts')

    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, total_steps=4000):
            super(CustomSchedule, self).__init__()

            self.d_model = d_model
            self.d_model = tf.cast(self.d_model, tf.float32)
            self.total_step = total_steps
            self.warmup_steps = total_steps*0.10

        def __call__(self, step):
            arg1 = step/self.warmup_steps
            arg2 = 1-(step-self.warmup_steps)/(self.total_step-self.warmup_steps)

            return 10e-5* tf.math.minimum(arg1, arg2)

    steps_per_epoch = len(train_dataset)
    learning_rate = CustomSchedule(128,100*steps_per_epoch)
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    
    value_range = graph_dataset.value_range 

    mse = 10000
    stopping_monitor = 0
    for epoch in range(200):
        mse_object = tf.keras.metrics.MeanSquaredError()
        for x,adjoin_matrix,y in train_dataset:
            with tf.GradientTape() as tape:
                seq = tf.cast(tf.math.equal(x, 0), tf.float32)
                mask = seq[:, tf.newaxis, tf.newaxis, :]
                preds = model(x,mask=mask,training=True,adjoin_matrix=adjoin_matrix)
                loss = tf.reduce_mean(tf.square(y-preds))
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                mse_object.update_state(y,preds)
        print('epoch: ',epoch,'loss: {:.4f}'.format(loss.numpy().item()))

        y_true = []
        y_preds = []
        for x, adjoin_matrix, y in val_dataset:
            seq = tf.cast(tf.math.equal(x, 0), tf.float32)
            mask = seq[:, tf.newaxis, tf.newaxis, :]
            preds = model(x,mask=mask,adjoin_matrix=adjoin_matrix,training=False)
            y_true.append(y.numpy())
            y_preds.append(preds.numpy())
        y_true = np.concatenate(y_true,axis=0).reshape(-1)
        y_preds = np.concatenate(y_preds,axis=0).reshape(-1)
        mse_new = keras.metrics.MSE(y_true, y_preds).numpy() * (value_range**2)
        val_mse = keras.metrics.MSE(y_true, y_preds).numpy() * (value_range**2)
        print(f'val mse: {val_mse.item()}')
        if mse_new.item() < mse:
            mse = mse_new.item()
            stopping_monitor = 0
            np.save('{}/{}{}{}{}{}'.format(arch['path'], task, seed, arch['name'], trained_epoch, trained_epoch,pretraining_str),
                    [y_true, y_preds])
            model.save_weights('regression_weights/{}.h5'.format(task))
        else:
            stopping_monitor +=1
        print('best mse: {:.4f}'.format(mse))
        if stopping_monitor>0:
            print('stopping_monitor:',stopping_monitor)
        if stopping_monitor>30:
            break

    y_true = []
    y_preds = []
    model.load_weights('regression_weights/{}.h5'.format(task, seed))
    for x, adjoin_matrix, y in test_dataset:
        seq = tf.cast(tf.math.equal(x, 0), tf.float32)
        mask = seq[:, tf.newaxis, tf.newaxis, :]
        preds = model(x, mask=mask, adjoin_matrix=adjoin_matrix, training=False)
        y_true.append(y.numpy())
        y_preds.append(preds.numpy())

    y_true = np.concatenate(y_true, axis=0).reshape(-1,len(label))
    y_preds = np.concatenate(y_preds, axis=0).reshape(-1,len(label))


    test_mse = keras.metrics.MSE(y_true.reshape(-1), y_preds.reshape(-1)).numpy() * (value_range**2)
    test_RMSE = np.sqrt(test_mse.item())
    print('test rmse:{:.4f}'.format(test_RMSE))

    return mse, test_RMSE

if __name__ == "__main__":
    mse_list = []
    RMSE_list = []
    for seed in [1,2,3]:
        print(seed)
        args = {"dense_dropout":0.05, "learning_rate":0.0000636859, "batch_size":16, "num_heads":8}
        MSE, test_RMSE= main(seed, args)
        mse_list.append(MSE)
        RMSE_list.append(test_RMSE)
    mse_list.append(np.mean(mse_list))
    RMSE_list.append(np.mean(RMSE_list))
    print(mse_list)
    print(RMSE_list)
