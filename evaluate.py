from nlgeval import compute_metrics
from path_datasets_and_weights import *
from setting import *
EPOCHS,BATCH_SIZE,MILESTONES,DATASET_PATH = Train_setting(DATASET_NAME,PHASE)
SOURCES, TARGETS, KW_SRC, KW_TGT, KW_OUT = input_choosing(MODEL_NAME)
    
dataset, all_data, train_data, val_data, test_data, VOCAB_SIZE, POSIT_SIZE, NUM_LABELS, NUM_CLASSES, COMMENT = Dataset_setting(DATASET_NAME,DATASET_PATH,IMAGE_MODE,SOURCES,TARGETS)
    
metrics_dict = compute_metrics(hypothesis='outputs/x_{}_{}_{}_{}_{}_Hyp.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT, IMAGE_MODE),
                               references=['outputs/x_{}_{}_{}_{}_Ref.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT)])

# metrics_dict = compute_metrics(hypothesis='outputs/x_FFAIR_VisualTransformer_ResNet50_MaxView2_NumLabel113_NoHistory_Hyp.txt',
#                                references=['outputs/x_FFAIR_VisualTransformer_ResNet50_MaxView2_NumLabel113_NoHistory_Ref.txt'])

print(metrics_dict)
