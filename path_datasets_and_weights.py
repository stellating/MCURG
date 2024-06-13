"""
NLMCXR dataset is downloaded from https://openi.nlm.nih.gov/faq#collection.
Images are put into the path: 
/home/tianyan/datasets
    |--NLMCXR
        |-cxr_images
            |-CXR1_1_IM-0001-3001.png
            |-....png
        |-ecgen-radiology
            |-1.xml
            |-....xml


"""

RELOAD = True # True / False
PHASE = 'TRAIN' # TRAIN / TEST / INFER / FEATURE
DATASET_NAME = 'NLMCXR' # NIHCXR / NLMCXR / MIMIC 
BACKBONE_NAME = 'DenseNet121' # ResNeSt50 / ResNet50 / DenseNet121
MODEL_NAME = 'ClsGenInt' # ClsGen / ClsGenInt / VisualTransformer / GumbelTransformer
IMAGE_MODE='unbias' # mask / unbias / origin
checkpoint_path_from = ['/root/autodl-tmp/NLMCXR_ClsGenInt_DenseNet121_MaxView2_NumLabel114_History(1).pt','checkpoints/NLMCXR_ClsGenInt_DenseNet121_mask_0527.pt']#.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME)
# checkpoint_path_from = 'checkpoints/{}_{}_{}_{}_300.pt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, IMAGE_MODE)
# checkpoint_path_to = 'checkpoints/unbias_.pt'
# checkpoint_path_from = '/root/autodl-tmp/NLMCXR_ClsGenInt_DenseNet121_MaxView2_NumLabel114_History(1).pt'
checkpoint_path_to = 'checkpoints/{}_{}_{}_{}_0528-remove2.pt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, IMAGE_MODE)
LOG_PATH='../tf-logs/'