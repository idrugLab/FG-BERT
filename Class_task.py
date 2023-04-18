import sklearn
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from dataset_scoffold_random import Graph_Classification_Dataset
import os
from muti_model import PredictModel, BertModel
from sklearn.metrics import r2_score, roc_auc_score
from hyperopt import fmin, tpe, hp
from utils import get_task_names
import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
keras.backend.clear_session()
os.environ['TF_DETERMINISTIC_OPS'] = '1'


def main(seed, args):

    # tasks = ['BBBP', 'bace', 'HIV','clintox', 'tox21', 'muv', 'sider','']

    task = 'BBBP'

    if task == 'BBBP':
        label = ['p_np']

    elif task =='bace':
        label = ['Class']

    elif task == 'HIV':
        label = ['HIV_active']

    elif task == 'clintox':
        label = ['FDA_APPROVED', 'CT_TOX']

    elif task == 'tox21':
        label = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
        
    elif task == 'muv':
        label = ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652',	'MUV-689', 'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810', 'MUV-832',	'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859'
    ]
        
    elif task == 'sider':
        label = ['Hepatobiliary disorders','Metabolism and nutrition disorders', 'Product issues', 'Eye disorders', 'Investigations','Musculoskeletal and connective tissue disorders',
        'Gastrointestinal disorders', 'Social circumstances', 'Immune system disorders', 'Reproductive system and breast disorders', 'Neoplasms benign, malignant and unspecified (incl cysts and polyps)', 
        'General disorders and administration site conditions', 'Endocrine disorders', 'Surgical and medical procedures', 'Vascular disorders', 'Blood and lymphatic system disorders', 
        'Skin and subcutaneous tissue disorders', 'Congenital, familial and genetic disorders', 'Infections and infestations', 'Respiratory, thoracic and mediastinal disorders', 'Psychiatric disorders', 
        'Renal and urinary disorders', 'Pregnancy, puerperium and perinatal conditions', 'Ear and labyrinth disorders', 'Cardiac disorders', 'Nervous system disorders', 'Injury, poisoning and procedural complications'
    ]

    elif task == 'toxcast_data':
        label = get_task_names('toxcast_data.csv')
    
    elif task == 'estrogen':
        label = ['alpha','beta']

    arch = {'name': 'Medium', 'path': 'medium3_weights_20_notihuan_BBBP'}
    pretraining = True
    pretraining_str = 'pretraining' if pretraining else ''
    trained_epoch = 20
    num_layers = 6
    d_model = 256
    addH = True
    dff = d_model * 2
    vocab_size = 18

    num_heads = args['num_heads']
    dense_dropout = args['dense_dropout']
    learning_rate = args['learning_rate']
    batch_size = args['batch_size']
    seed = seed
    np.random.seed(seed=seed)
    tf.random.set_seed(seed=seed)
    train_dataset, test_dataset, val_dataset = Graph_Classification_Dataset('BBBP.csv', smiles_field='smiles',
                                                           label_field=label, seed=seed,batch_size=batch_size,a = len(label), addH=True).get_data()
                                                        
    x, adjoin_matrix, y = next(iter(train_dataset.take(1)))
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]

    model = PredictModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,a=len(label),
                         dense_dropout = dense_dropout)

    if pretraining:
        temp = BertModel(num_layers=num_layers, d_model=d_model,
                         dff=dff, num_heads=num_heads, vocab_size=vocab_size)

        pred = temp(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)
        temp.load_weights(
            arch['path']+'/bert_weights{}_{}.h5'.format(arch['name'], trained_epoch))
        temp.encoder.save_weights(
            arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'], trained_epoch))
        del temp

        pred = model(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)
        model.encoder.load_weights(
            arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'], trained_epoch))
        print('load_wieghts')

    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

    auc = -10
    stopping_monitor = 0
    for epoch in range(200):
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        for x, adjoin_matrix, y in train_dataset:
            with tf.GradientTape() as tape:
                seq = tf.cast(tf.math.equal(x, 0), tf.float32)
                mask = seq[:, tf.newaxis, tf.newaxis, :]
                preds = model(x, mask=mask, training=True,adjoin_matrix=adjoin_matrix)
                loss = 0
                for i in range(len(label)):
                    y_label = y[:,i]
                    y_pred = preds[:,i]
                    validId = np.where((y_label == 0) | (y_label == 1))[0]
                    if len(validId) == 0:
                        continue
                    y_t = tf.gather(y_label,validId)
                    y_p = tf.gather(y_pred,validId)
            
                    loss += loss_object(y_t, y_p)
                loss = loss/(len(label))
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print('epoch: ', epoch, 'loss: {:.4f}'.format(loss.numpy().item()))

        y_true = {}
        y_preds = {}
        for i in range(len(label)):
            y_true[i] = []
            y_preds[i] = []
        for x, adjoin_matrix, y in val_dataset:
            seq = tf.cast(tf.math.equal(x, 0), tf.float32)
            mask = seq[:, tf.newaxis, tf.newaxis, :]
            preds = model(x, mask=mask, adjoin_matrix=adjoin_matrix, training=False)
            for i in range(len(label)):
                y_label = y[:,i]
                y_pred = preds[:,i]
                y_true[i].append(y_label)
                y_preds[i].append(y_pred)
        y_tr_dict = {}
        y_pr_dict = {}
        for i in range(len(label)):
            y_tr = np.array([])
            y_pr = np.array([])
            for j in range(len(y_true[i])):
                a = np.array(y_true[i][j])
                b = np.array(y_preds[i][j])
                y_tr = np.concatenate((y_tr,a))
                y_pr = np.concatenate((y_pr,b))
            y_tr_dict[i] = y_tr
            y_pr_dict[i] = y_pr

        AUC_list = []

        for i in range(len(label)):
            y_label = y_tr_dict[i]
            y_pred = y_pr_dict[i]
            validId = np.where((y_label== 0) | (y_label == 1))[0]
            if len(validId) == 0:
                continue
            y_t = tf.gather(y_label,validId)
            y_p = tf.gather(y_pred,validId)
            if all(target == 0 for target in y_t) or all(target == 1 for target in y_t):
                AUC = float('nan')
                AUC_list.append(AUC)
                continue
            y_p = tf.sigmoid(y_p).numpy()
            AUC_new = sklearn.metrics.roc_auc_score(y_t, y_p, average=None)

            AUC_list.append(AUC_new)
        auc_new = np.nanmean(AUC_list)

        print('val auc:{:.4f}'.format(auc_new))
        if auc_new> auc:
            auc = auc_new
            stopping_monitor = 0
            np.save('{}/{}{}{}{}{}'.format(arch['path'], task, seed, arch['name'], trained_epoch, trained_epoch, pretraining_str),
                    [y_true, y_preds])
            model.save_weights('classification_weights/{}_{}.h5'.format(task, seed))
            print('save model weights')
        else:
            stopping_monitor += 1
        print('best val auc: {:.4f}'.format(auc))
        if stopping_monitor > 0:
            print('stopping_monitor:', stopping_monitor)
        if stopping_monitor > 30:
            break

    y_true = {}
    y_preds = {}
    for i in range(len(label)):
        y_true[i] = []
        y_preds[i] = []
    model.load_weights('classification_weights/{}_{}.h5'.format(task, seed))
    for x, adjoin_matrix, y in test_dataset:
        seq = tf.cast(tf.math.equal(x, 0), tf.float32)
        mask = seq[:, tf.newaxis, tf.newaxis, :]
        preds = model(x, mask=mask, adjoin_matrix=adjoin_matrix,training=False)
        for i in range(len(label)):
            y_label = y[:,i]
            y_pred = preds[:,i]
            y_true[i].append(y_label)
            y_preds[i].append(y_pred)
    y_tr_dict = {}
    y_pr_dict = {}
    for i in range(len(label)):
        y_tr = np.array([])
        y_pr = np.array([])
        for j in range(len(y_true[i])):
            a = np.array(y_true[i][j])
            if a.ndim == 0:
                continue
            b = np.array(y_preds[i][j])
            y_tr = np.concatenate((y_tr,a))
            y_pr = np.concatenate((y_pr,b))
        y_tr_dict[i] = y_tr
        y_pr_dict[i] = y_pr
    auc_list = []

    for i in range(len(label)):
        y_label = y_tr_dict[i]
        y_pred = y_pr_dict[i]
        validId = np.where((y_label== 0) | (y_label == 1))[0]
        if len(validId) == 0:
            continue
        y_t = tf.gather(y_label,validId)
        y_p = tf.gather(y_pred,validId)
        if all(target == 0 for target in y_t) or all(target == 1 for target in y_t):
            AUC = float('nan')
            auc_list.append(AUC)
            continue
        y_p = tf.sigmoid(y_p).numpy()
        AUC_new = sklearn.metrics.roc_auc_score(y_t, y_p, average=None)
        auc_list.append(AUC_new)
    test_auc = np.nanmean(auc_list)

    print('test auc:{:.4f}'.format(test_auc))

    return auc, test_auc, auc_list                                                                

if __name__ == '__main__':

    args = {"dense_dropout":0.1, "learning_rate":1e-4 , "batch_size":16, "num_heads":4}
    auc_list = []
    test_auc_list = []
    test_all_auc_list = []
    for seed in [1,2,3]:
        print(seed)
        auc, test_auc, a_list= main(seed, args)
        auc_list.append(auc)
        test_auc_list.append(test_auc)
        test_all_auc_list.append(a_list)
    auc_list.append(np.mean(auc_list))
    test_auc_list.append(np.mean(test_auc_list))
    print(auc_list)
    print(test_auc_list)
    print(test_all_auc_list)
