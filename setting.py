"""
Set the modules according to different models and datasets.
"""
import torch
import torchvision.models as tmodels
import torch.nn as nn
from datasets import NIHCXR, MIMIC, NLMCXR
from losses import CELoss, CELossTotal, CELossTotalEval, CELossTransfer, CELossShift
from models import CNN, MVCNN, TNN, Classifier, Generator, ClsGen, ClsGenInt
from path_datasets_and_weights import (
    RELOAD,
    IMAGE_MODE
)

def Train_setting(DATASET_NAME,PHASE):
        
    if DATASET_NAME == 'MIMIC':
        EPOCHS = 50 # Start overfitting after 20 epochs
        BATCH_SIZE = 8 if PHASE == 'TRAIN' else 64 # 128 # Fit 4 GPUs
        MILESTONES = [1000] # Reduce LR by 10 after reaching milestone epochs
        DATASET_PATH = ''
        
    elif DATASET_NAME == 'NLMCXR':
        EPOCHS = 2000 # Start overfitting after 20 epochs
        BATCH_SIZE = 64 if PHASE == 'TRAIN' else 256  # Fit 4 GPUs
        MILESTONES = [100] # Reduce LR by 10 after reaching milestone epochs
        DATASET_PATH = '/root/autodl-tmp/NLMCXR/'
        
    else:
        raise ValueError('Invalid DATASET_NAME')
    return EPOCHS,BATCH_SIZE,MILESTONES,DATASET_PATH


def input_choosing(MODEL_NAME,PHASE=None):
    # --- Choose Inputs/Outputs
    if MODEL_NAME in ['ClsGen', 'ClsGenInt']:
        SOURCES = ['image','caption','label','history']
        TARGETS = ['caption','label']
        KW_SRC = ['image','caption','label','history']
        KW_TGT = None
        KW_OUT = None
                
    elif MODEL_NAME == 'VisualTransformer':
        SOURCES = ['image','caption']
        TARGETS = ['caption']# ,'label']
        KW_SRC = ['image','caption'] # kwargs of Classifier
        KW_TGT = None
        KW_OUT = None
        
    elif MODEL_NAME == 'GumbelTransformer':
        SOURCES = ['image','caption','caption_length']
        TARGETS = ['caption','label']
        KW_SRC = ['image','caption','caption_length'] # kwargs of Classifier
        KW_TGT = None
        KW_OUT = None
        
    else:
        raise ValueError('Invalid BACKBONE_NAME')
    
    if PHASE == 'FEATURE':
        SOURCES += ['image_name']
    return SOURCES, TARGETS, KW_SRC, KW_TGT, KW_OUT

def Dataset_setting(DATASET_NAME, DATASET_PATH, IMAGE_MODE, SOURCES, TARGETS):
    # --- Choose a Dataset ---
    if DATASET_NAME == 'MIMIC':
        INPUT_SIZE = (256,256)
        MAX_VIEWS = 2
        NUM_LABELS = 114
        NUM_CLASSES = 2
        
        dataset = MIMIC('/home/hoang/Datasets/MIMIC/', INPUT_SIZE, view_pos=['AP','PA','LATERAL'], max_views=MAX_VIEWS, sources=SOURCES, targets=TARGETS)
        train_data, val_data, test_data = dataset.get_subsets(pvt=0.9, seed=0, generate_splits=True, debug_mode=False, train_phase=(PHASE == 'TRAIN'))
        
        VOCAB_SIZE = len(dataset.vocab)
        POSIT_SIZE = dataset.max_len
        COMMENT = 'MaxView{}_NumLabel{}_{}History'.format(MAX_VIEWS, NUM_LABELS, 'No' if 'history' not in SOURCES else '')
            
    elif DATASET_NAME == 'NLMCXR':
        INPUT_SIZE = (256,256)
        MAX_VIEWS = 2
        NUM_LABELS = 114
        NUM_CLASSES = 2

        dataset = NLMCXR(DATASET_PATH, IMAGE_MODE, INPUT_SIZE, view_pos=['AP','PA','LATERAL'], max_views=MAX_VIEWS, sources=SOURCES, targets=TARGETS)
        all_data, train_data, val_data, test_data = dataset.get_subsets(IMAGE_MODE, seed=123)
        
        VOCAB_SIZE = len(dataset.vocab)
        POSIT_SIZE = dataset.max_len
        COMMENT = 'MaxView{}_NumLabel{}_{}History'.format(MAX_VIEWS, NUM_LABELS, 'No' if 'history' not in SOURCES else '')
        
    else:
        raise ValueError('Invalid DATASET_NAME')
    
    return dataset, all_data, train_data, val_data, test_data, VOCAB_SIZE, POSIT_SIZE, NUM_LABELS, NUM_CLASSES, COMMENT

def Backbone_setting(BACKBONE_NAME):
    # --- Choose a Backbone --- 
    if BACKBONE_NAME == 'ResNeSt50':
        torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
        backbone = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', weights="IMAGENET1K_V1")
        FC_FEATURES = 2048
        
    elif BACKBONE_NAME == 'ResNet50':
        backbone = tmodels.resnet50(weights = "IMAGENET1K_V1")
        FC_FEATURES = 2048
        
    elif BACKBONE_NAME == 'DenseNet121':
        # 实例化 DenseNet121 模型
        backbone = tmodels.densenet121(pretrained=True)
        # backbone = torch.hub.load('pytorch/vision', 'densenet121', weights="IMAGENET1K_V1")
        # backbone = torch.hub.load('densenet121-a639ec97.pth','densenet121')
        FC_FEATURES = 1024
        
    else:
        raise ValueError('Invalid BACKBONE_NAME')
    
    return backbone, FC_FEATURES

def Model_setting(MODEL_NAME,backbone,BACKBONE_NAME,VOCAB_SIZE,POSIT_SIZE,NUM_LABELS, NUM_CLASSES, FC_FEATURES):
    # --- Choose a Model ---
    if MODEL_NAME == 'ClsGen':
        LR = 1e-6 # Fastest LR
        WD = 1e-2 # Avoid overfitting with L2 regularization
        DROPOUT = 0.1 # Avoid overfitting
        NUM_EMBEDS = 256
        FWD_DIM = 256
        HIDDEN_SIZE = None
        
        NUM_HEADS = 8
        NUM_LAYERS = 1
        
        cnn = CNN(backbone, BACKBONE_NAME)
        cnn = MVCNN(cnn)
        tnn = TNN(embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, dropout=DROPOUT, num_layers=NUM_LAYERS, num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE)
        
        # Not enough memory to run 8 heads and 12 layers, instead 1 head is enough
        NUM_HEADS = 1
        NUM_LAYERS = 12
        
        cls_model = Classifier(num_topics=NUM_LABELS, num_states=NUM_CLASSES, cnn=cnn, tnn=tnn, fc_features=FC_FEATURES, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, dropout=DROPOUT)
        gen_model = Generator(num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, dropout=DROPOUT, num_layers=NUM_LAYERS)
        
        model = ClsGen(cls_model, gen_model, NUM_LABELS, NUM_EMBEDS)
        criterion = CELossTotal(ignore_index=3)
        
    elif MODEL_NAME == 'ClsGenInt':
        LR = 3e-4 # Slower LR to fine-tune the model (Open-I)
        # LR = 3e-6 # Slower LR to fine-tune the model (MIMIC)
        WD = 1e-2 # Avoid overfitting with L2 regularization
        DROPOUT = 0.1 # Avoid overfitting
        NUM_EMBEDS = 256
        FWD_DIM = 256
        
        NUM_HEADS = 8
        NUM_LAYERS = 1
        HIDDEN_SIZE = 0
        
        cnn = CNN(backbone, BACKBONE_NAME)
        cnn = MVCNN(cnn)
        tnn = TNN(embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, dropout=DROPOUT, num_layers=NUM_LAYERS, num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE)
        
        # Not enough memory to run 8 heads and 12 layers, instead 1 head is enough
        NUM_HEADS = 1
        NUM_LAYERS = 12

        if IMAGE_MODE == 'unbias':
            cls_model_origin = Classifier(num_topics=NUM_LABELS, num_states=NUM_CLASSES, cnn=cnn, tnn=tnn, fc_features=FC_FEATURES, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, dropout=DROPOUT)
            cls_model_mask = Classifier(num_topics=NUM_LABELS, num_states=NUM_CLASSES, cnn=cnn, tnn=tnn, fc_features=FC_FEATURES, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, dropout=DROPOUT)
            cls_model = [cls_model_origin,cls_model_mask]
            gen_model = Generator(num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, dropout=DROPOUT, num_layers=NUM_LAYERS)
            
            clsgen_model = ClsGen(cls_model, gen_model, NUM_LABELS, NUM_EMBEDS)
            clsgen_model = nn.DataParallel(clsgen_model).cuda()
        else:
            cls_model = Classifier(num_topics=NUM_LABELS, num_states=NUM_CLASSES, cnn=cnn, tnn=tnn, fc_features=FC_FEATURES, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, dropout=DROPOUT)
            gen_model = Generator(num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, dropout=DROPOUT, num_layers=NUM_LAYERS)
            
            clsgen_model = ClsGen(cls_model, gen_model, NUM_LABELS, NUM_EMBEDS)
            clsgen_model = nn.DataParallel(clsgen_model).cuda()
        
        
        # Initialize the Interpreter module
        NUM_HEADS = 8
        NUM_LAYERS = 1
        
        tnn = TNN(embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, dropout=DROPOUT, num_layers=NUM_LAYERS, num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE)
        int_model = Classifier(num_topics=NUM_LABELS, num_states=NUM_CLASSES, cnn=None, tnn=tnn, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, dropout=DROPOUT)
        int_model = nn.DataParallel(int_model).cuda()
        
        
        model = ClsGenInt(clsgen_model.module.cpu(), int_model.module.cpu(), freeze_evaluator=True)
        criterion = CELossTotalEval(ignore_index=3)
        
    elif MODEL_NAME == 'VisualTransformer':
        # Clinical Coherent X-ray Report (Justin et. al.) - No Fine-tune
        LR = 5e-5
        WD = 1e-2 # Avoid overfitting with L2 regularization
        DROPOUT = 0.1 # Avoid overfitting
        NUM_EMBEDS = 256
        NUM_HEADS = 8
        FWD_DIM = 4096
        NUM_LAYERS_ENC = 1
        NUM_LAYERS_DEC = 6
        
        cnn = CNN(backbone, BACKBONE_NAME)
        cnn = MVCNN(cnn)
        model = Transformer(image_encoder=cnn, num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE, 
                            fc_features=FC_FEATURES, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, 
                            dropout=DROPOUT, num_layers_enc=NUM_LAYERS_ENC, num_layers_dec=NUM_LAYERS_DEC, freeze_encoder=True)
        criterion = CELossShift(ignore_index=3)
        
    elif MODEL_NAME == 'GumbelTransformer':
        # Clinical Coherent X-ray Report (Justin et. al.)        
        LR = 5e-5
        WD = 1e-2 # Avoid overfitting with L2 regularization
        DROPOUT = 0.1 # Avoid overfitting
        NUM_EMBEDS = 256
        NUM_HEADS = 8
        FWD_DIM = 4096
        NUM_LAYERS_ENC = 1
        NUM_LAYERS_DEC = 6
        
        cnn = CNN(backbone, BACKBONE_NAME)
        cnn = MVCNN(cnn)
        transformer = Transformer(image_encoder=cnn, num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE, 
                                  fc_features=FC_FEATURES, embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, 
                                  dropout=DROPOUT, num_layers_enc=NUM_LAYERS_ENC, num_layers_dec=NUM_LAYERS_DEC, freeze_encoder=True)
        transformer = nn.DataParallel(transformer).cuda()
        pretrained_from = 'checkpoints/{}_{}_{}_{}.pt'.format(DATASET_NAME,'VisualTransformer',BACKBONE_NAME,COMMENT)
        last_epoch, (best_metric, test_metric) = load(pretrained_from, transformer)
        print('Reload From: {} | Last Epoch: {} | Validation Metric: {} | Test Metric: {}'.format(pretrained_from, last_epoch, best_metric, test_metric))
        
        NUM_EMBEDS = 256
        HIDDEN_SIZE = 128
        
        pretrained_from = 'checkpoints/{}_{}_{}_NumLabel{}.pt'.format(DATASET_NAME,'LSTM','MaxView2',NUM_LABELS)
        diff_chexpert = LSTM_Attn(num_tokens=VOCAB_SIZE, embed_dim=NUM_EMBEDS, hidden_size=HIDDEN_SIZE, num_topics=NUM_LABELS, num_states=NUM_CLASSES, dropout=DROPOUT)
        diff_chexpert = nn.DataParallel(diff_chexpert).cuda()
        last_epoch, (best_metric, test_metric) = load(pretrained_from, diff_chexpert)
        print('Reload From: {} | Last Epoch: {} | Validation Metric: {} | Test Metric: {}'.format(pretrained_from, last_epoch, best_metric, test_metric))
        
        model = GumbelTransformer(transformer.module.cpu(), diff_chexpert.module.cpu())
        criterion = CELossTotal(ignore_index=3)
        
    elif MODEL_NAME == 'ST':
        KW_SRC = ['image', 'caption', 'caption_length']
        
        LR = 5e-5
        NUM_EMBEDS = 256
        HIDDEN_SIZE = 128
        DROPOUT = 0.1
        
        model = ST(cnn, num_tokens=VOCAB_SIZE, fc_features=FC_FEATURES, embed_dim=NUM_EMBEDS, hidden_size=HIDDEN_SIZE, dropout=DROPOUT, freeze_encoder=True)
        criterion = CELossShift(ignore_index=3)
        
    else:
        raise ValueError('Invalid MODEL_NAME')
    return model, criterion, LR, WD, NUM_EMBEDS, HIDDEN_SIZE, DROPOUT