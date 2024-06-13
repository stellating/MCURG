# --- Base packages ---
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

# --- PyTorch packages ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# --- Helper Packages ---
from tqdm import tqdm

# --- Project Packages ---
from utils import save, load, unbias_load, train, test, data_to_device, data_concatenate
from datasets import NIHCXR, MIMIC, NLMCXR
from losses import CELoss, CELossTotal, CELossTotalEval, CELossTransfer, CELossShift
from models import CNN, MVCNN, TNN, Classifier, Generator, ClsGen, ClsGenInt
from baselines.transformer.models import LSTM_Attn, Transformer, GumbelTransformer
from baselines.rnn.models import ST
from path_datasets_and_weights import (
    RELOAD,
    PHASE,
    DATASET_NAME,
    BACKBONE_NAME,
    MODEL_NAME,
    IMAGE_MODE,
    checkpoint_path_from,
    checkpoint_path_to,
    LOG_PATH
)
from torch.utils.tensorboard import SummaryWriter   
from setting import *

# --- Helper Functions ---
def find_optimal_cutoff(target, predicted):
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    return threshold[ix]

def infer(data_loader, model, device='cpu', threshold=None):
    model.eval()
    outputs = []
    targets = []

    with torch.no_grad():
        prog_bar = tqdm(data_loader)
        for i, (source, target) in enumerate(prog_bar):
            source = data_to_device(source, device)
            target = data_to_device(target, device)

            # Use single input if there is no clinical history
            if threshold != None:
                output = model(image=source[0], history=source[3], threshold=threshold)
                # output = model(image=source[0], threshold=threshold)
                # output = model(image=source[0], history=source[3], label=source[2])
                # output = model(image=source[0], label=source[2])
            else:
                # output = model(source[0], source[1])
                output = model(source[0])
            print(output)
                
            outputs.append(data_to_device(output))
            targets.append(data_to_device(target))

        outputs = data_concatenate(outputs)
        targets = data_concatenate(targets)
    
    return outputs, targets

# --- Hyperparameters ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.manual_seed(seed=123)



if __name__ == "__main__":
    EPOCHS,BATCH_SIZE,MILESTONES,DATASET_PATH = Train_setting(DATASET_NAME,PHASE)
    SOURCES, TARGETS, KW_SRC, KW_TGT, KW_OUT = input_choosing(MODEL_NAME)
    dataset, all_data, train_data, val_data, test_data, VOCAB_SIZE, POSIT_SIZE, NUM_LABELS, NUM_CLASSES, COMMENT = Dataset_setting(DATASET_NAME,DATASET_PATH,IMAGE_MODE,SOURCES,TARGETS)
    backbone, FC_FEATURES = Backbone_setting(BACKBONE_NAME)
    model, criterion, LR, WD, NUM_EMBEDS, HIDDEN_SIZE, DROPOUT = Model_setting(MODEL_NAME, backbone, BACKBONE_NAME, VOCAB_SIZE, POSIT_SIZE, NUM_LABELS,NUM_CLASSES,FC_FEATURES) 
    writer = SummaryWriter(LOG_PATH)
    print("Finished all the settings!")
    # --- Main program ---
    train_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, drop_last=True)
    val_loader = data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    test_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    model = model.cuda()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES)

    print('Total Parameters:', sum(p.numel() for p in model.parameters()))
    
    
    last_epoch = -1
    best_metric = 1e9

    # checkpoint_path_from = 'checkpoints/{}_{}_{}_{}.pt'.format(DATASET_NAME,MODEL_NAME,BACKBONE_NAME,COMMENT)
    # checkpoint_path_to = 'checkpoints/{}_{}_{}_{}.pt'.format(DATASET_NAME,MODEL_NAME,BACKBONE_NAME,COMMENT)
    
    if RELOAD:
        if IMAGE_MODE=='unbias':
            if PHASE=='TRAIN':
                last_epoch, (best_metric, test_metric) = unbias_load(checkpoint_path_from, model, optimizer, scheduler) # Reload
                print('Reload From: {} | Last Epoch: {} | Validation Metric: {} | Test Metric: {}'.format(checkpoint_path_from[0], last_epoch, best_metric, test_metric)) 
            else:
                last_epoch, (best_metric, test_metric) = load(checkpoint_path_from, model, optimizer, scheduler) # Reload
                print('Reload From: {} | Last Epoch: {} | Validation Metric: {} | Test Metric: {}'.format(checkpoint_path_from, last_epoch, best_metric, test_metric))

        else:
            last_epoch, (best_metric, test_metric) = load(checkpoint_path_from, model, optimizer, scheduler) # Reload
            # last_epoch, (best_metric, test_metric) = load(checkpoint_path_from, model) # Fine-tune
            print('Reload From: {} | Last Epoch: {} | Validation Metric: {} | Test Metric: {}'.format(checkpoint_path_from, last_epoch, best_metric, test_metric))

    if PHASE == 'TRAIN':
        scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(last_epoch+1, EPOCHS):
            print('Epoch:', epoch)
            train_loss = train(train_loader, model, optimizer, criterion, writer, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT, scaler=scaler)
            val_loss = test(val_loader, model, criterion, writer, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT, return_results=False)
            test_loss = test(test_loader, model, criterion, writer, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT, return_results=False)
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('test_loss', test_loss, epoch)
            
            scheduler.step()
            
            if best_metric > val_loss:
                best_metric = val_loss
                save(checkpoint_path_to, model, optimizer, scheduler, epoch, (val_loss, test_loss))
                print('New Best Metric: {}'.format(best_metric)) 
                print('Saved To:', checkpoint_path_to)
    
    elif PHASE == 'TEST':
        # Output the file list for inspection
        out_file_img = open('outputs/{}_{}_{}_{}_Img.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT), 'w')
        for i in range(len(test_data.idx_pidsid)):
            out_file_img.write(test_data.idx_pidsid[i][0] + ' ' + test_data.idx_pidsid[i][1] + '\n')
            
        # test_loss, test_outputs, test_targets = test(test_loader, model, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT, select_outputs=[1])
        
        # test_auc = []
        # test_f1 = []
        # test_prc = []
        # test_rec = []
        # test_acc = []
        
        # threshold = 0.25
        # NUM_LABELS = 14
        # for i in range(NUM_LABELS):
        #     try:
        #         test_auc.append(metrics.roc_auc_score(test_targets.cpu()[...,i], test_outputs.cpu()[...,i,1]))
        #         test_f1.append(metrics.f1_score(test_targets.cpu()[...,i], test_outputs.cpu()[...,i,1] > threshold))
        #         test_prc.append(metrics.precision_score(test_targets.cpu()[...,i], test_outputs.cpu()[...,i,1] > threshold))
        #         test_rec.append(metrics.recall_score(test_targets.cpu()[...,i], test_outputs.cpu()[...,i,1] > threshold))
        #         test_acc.append(metrics.accuracy_score(test_targets.cpu()[...,i], test_outputs.cpu()[...,i,1] > threshold))
                
        #     except:
        #         print('An error occurs for label', i)
                
        # test_auc = np.mean([x for x in test_auc if str(x) != 'nan'])
        # test_f1 = np.mean([x for x in test_f1 if str(x) != 'nan'])
        # test_prc = np.mean([x for x in test_prc if str(x) != 'nan'])
        # test_rec = np.mean([x for x in test_rec if str(x) != 'nan'])
        # test_acc = np.mean([x for x in test_acc if str(x) != 'nan'])
        
        # print('Accuracy       : {}'.format(test_acc))
        # print('Macro AUC      : {}'.format(test_auc))
        # print('Macro F1       : {}'.format(test_f1))
        # print('Macro Precision: {}'.format(test_prc))
        # print('Macro Recall   : {}'.format(test_rec))
        # print('Micro AUC      : {}'.format(metrics.roc_auc_score(test_targets.cpu()[...,:NUM_LABELS] == 1, test_outputs.cpu()[...,:NUM_LABELS,1], average='micro')))
        # print('Micro F1       : {}'.format(metrics.f1_score(test_targets.cpu()[...,:NUM_LABELS] == 1, test_outputs.cpu()[...,:NUM_LABELS,1] > threshold, average='micro')))
        # print('Micro Precision: {}'.format(metrics.precision_score(test_targets.cpu()[...,:NUM_LABELS] == 1, test_outputs.cpu()[...,:NUM_LABELS,1] > threshold, average='micro')))
        # print('Micro Recall   : {}'.format(metrics.recall_score(test_targets.cpu()[...,:NUM_LABELS] == 1, test_outputs.cpu()[...,:NUM_LABELS,1] > threshold, average='micro')))
        
#     elif PHASE == 'INFER':
#         txt_test_outputs, txt_test_targets = infer(test_loader, model, device='cuda', threshold=0.25)
#         gen_outputs = txt_test_outputs[0]
#         gen_targets = txt_test_targets[0]
#         print(gen_outputs,gen_targets)
        
#         out_file_ref = open('outputs/x_{}_{}_{}_{}_{}_300_Ref.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT, IMAGE_MODE), 'w')
#         out_file_hyp = open('outputs/x_{}_{}_{}_{}_{}_300_Hyp.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT, IMAGE_MODE), 'w')
#         out_file_lbl = open('outputs/x_{}_{}_{}_{}_{}_Lbl.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT,IMAGE_MODE), 'w')
#         # print('generated outputs: ',gen_outputs)
#         for i in range(len(gen_outputs)):  #遍历每句话
#             candidate = ''
#             for j in range(len(gen_outputs[i])):  # 遍历每个词
#                 tok = dataset.vocab.id_to_piece(int(gen_outputs[i,j]))
#                 if tok == '</s>':
#                     break # Manually stop generating token after </s> is reached
#                 elif tok == '<s>':
#                     continue
#                 elif tok == '▁': # space
#                     if len(candidate) and candidate[-1] != ' ':
#                         candidate += ' '
#                 elif tok in [',', '.', '-', ':']: # or not tok.isalpha():
#                     if len(candidate) and candidate[-1] != ' ':
#                         candidate += ' ' + tok + ' ' 
#                     else:
#                         candidate += tok + ' '
#                 else: # letter
#                     print(tok)
#                     candidate += tok + ' '
#             out_file_hyp.write(candidate + '\n')
            
#             reference = ''
#             for j in range(len(gen_targets[i])):
#                 tok = dataset.vocab.id_to_piece(int(gen_targets[i,j]))
#                 if tok == '</s>':
#                     break
#                 elif tok == '<s>':
#                     continue
#                 elif tok == '▁': # space
#                     if len(reference) and reference[-1] != ' ':
#                         reference += ' '
#                 elif tok in [',', '.', '-', ':']: # or not tok.isalpha():
#                     if len(reference) and reference[-1] != ' ':
#                         reference += ' ' + tok + ' ' 
#                     else:
#                         reference += tok + ' '
#                 else: # letter
#                     reference += tok    
#             out_file_ref.write(reference + '\n')

#         for i in tqdm(range(len(test_data))):
#             target = test_data[i][1] # caption, label
#             out_file_lbl.write(' '.join(map(str,target[1])) + '\n')
                
#     else:
#         raise ValueError('Invalid PHASE')

    elif PHASE == 'INFER':
        txt_test_outputs, txt_test_targets = infer(test_loader, model, device='cuda', threshold=0.25)
        gen_outputs = txt_test_outputs[0]
        gen_targets = txt_test_targets[0]
        
        out_file_ref = open('outputs/x_{}_{}_{}_{}_Ref.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT), 'w')
        out_file_hyp = open('outputs/x_{}_{}_{}_{}_{}_Hyp.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT, IMAGE_MODE), 'w')
        out_file_lbl = open('outputs/x_{}_{}_{}_{}_Lbl.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT), 'w')
        
        for i in range(len(gen_outputs)):
            candidate = ''
            for j in range(len(gen_outputs[i])):
                tok = dataset.vocab.id_to_piece(int(gen_outputs[i,j]))
                if tok == '</s>':
                    break # Manually stop generating token after </s> is reached
                elif tok == '<s>':
                    continue
                elif tok == '▁': # space
                    if len(candidate) and candidate[-1] != ' ':
                        candidate += ' '
                elif tok in [',', '.', '-', ':']: # or not tok.isalpha():
                    if len(candidate) and candidate[-1] != ' ':
                        candidate += ' ' + tok + ' ' 
                    else:
                        candidate += tok + ' '
                else: # letter
                    candidate += tok       
            out_file_hyp.write(candidate + '\n')
            
            reference = ''
            for j in range(len(gen_targets[i])):
                tok = dataset.vocab.id_to_piece(int(gen_targets[i,j]))
                if tok == '</s>':
                    break
                elif tok == '<s>':
                    continue
                elif tok == '▁': # space
                    if len(reference) and reference[-1] != ' ':
                        reference += ' '
                elif tok in [',', '.', '-', ':']: # or not tok.isalpha():
                    if len(reference) and reference[-1] != ' ':
                        reference += ' ' + tok + ' ' 
                    else:
                        reference += tok + ' '
                else: # letter
                    reference += tok    
            out_file_ref.write(reference + '\n')

        for i in tqdm(range(len(test_data))):
            target = test_data[i][1] # caption, label
            out_file_lbl.write(' '.join(map(str,target[1])) + '\n')
                
    else:
        raise ValueError('Invalid PHASE')