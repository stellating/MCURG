# --- Base packages ---
import os
import json
import numpy as np
import pandas as pd
import cv2

# --- PyTorch packages ---
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

# --- Helper packages ---
from random import shuffle
import sentencepiece as spm
from PIL import Image, ImageFile
import pickle as pkl
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Datasets ---
class NIHCXR(data.Dataset): # Chest X-Ray 14 Dataset
    def __init__(self, directory, input_size=(512,512), random_transform=True):
        self.list_diseases = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        self.dict_diseases = dict(zip(self.list_diseases, range(len(self.list_diseases))))

        self.dir = directory
        self.input_size = input_size
        self.random_transform = random_transform
        self.__input_data()
        
        if random_transform:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.1,0.1,0.1), 
                    transforms.RandomRotation(15, expand=True)]),
                transforms.Resize(input_size),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([transforms.Resize(input_size), transforms.ToTensor()])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(self.dir + 'images/' + self.img_files[idx]).convert('RGB')
        return self.transform(img), self.img_labels[idx]

    def __input_data(self):
        txt_file = self.dir + 'Data_Entry_2017_v2020.csv'
        data = np.loadtxt(open(txt_file, "rb"), delimiter=",", skiprows=1, dtype=str)
        self.img_files = data[..., 0]
        self.img_labels = self.__one_hot_outer(data[..., 1])

    def __one_hot_inner(self, labels):
        labels = labels.split('|')
        indices = []

        for label in labels:
            if label in self.dict_diseases:
                indices.append(self.dict_diseases[label])
            else:
                # Filtering invalid labels
                index = np.argmax([label in disease for disease in self.list_diseases])
                indices.append(index.item())

        labels = np.zeros(len(self.list_diseases))
        labels[indices] = 1
        return labels

    def __one_hot_outer(self, labels):
        one_hot = []
        for i in range(labels.shape[0]):
            one_hot.append(self.__one_hot_inner(labels[i]))
        return np.array(one_hot)

    def get_subsets(self, pvt=0.9, seed=0):
        file_to_label = dict(zip(self.img_files, self.img_labels))

        train_files = np.loadtxt(self.dir + 'train_val_list.txt', dtype=str)
        train_labels = np.array([file_to_label[f] for f in train_files])

        test_files = np.loadtxt(self.dir + 'test_list.txt', dtype=str)
        test_labels = np.array([file_to_label[f] for f in test_files])

        np.random.seed(seed)
        indices = np.random.permutation(len(train_files))
        pivot = int(len(train_files) * pvt)
        train_indices = indices[:pivot]
        val_indices = indices[pivot:]

        train_dataset = NIHCXR(self.dir, input_size=self.input_size, random_transform=self.random_transform)
        train_dataset.img_files = train_files[train_indices]
        train_dataset.img_labels = train_labels[train_indices]

        val_dataset = NIHCXR(self.dir, input_size=self.input_size, random_transform=False)
        val_dataset.img_files = train_files[val_indices]
        val_dataset.img_labels = train_labels[val_indices]

        test_dataset = NIHCXR(self.dir, input_size=self.input_size, random_transform=False)
        test_dataset.img_files = test_files
        test_dataset.img_labels = test_labels

        return train_dataset, val_dataset, test_dataset

class MIMIC(data.Dataset): # MIMIC-CXR Dataset
    def __init__(self, directory, input_size=(256,256), random_transform=True,
                view_pos=['AP', 'PA', 'LATERAL'], max_views=2, sources=['image','history'], targets=['label'], 
                max_len=1000, vocab_file='mimic_unigram_1000.model'):

        self.source_sections = ['INDICATION:', 'HISTORY:', 'CLINICAL HISTORY:', 'REASON FOR EXAM:', 'REASON FOR EXAMINATION:', 'CLINICAL INFORMATION:', 'CLINICAL INDICATION:', 'PATIENT HISTORY:']
        self.target_sections = ['FINDINGS:']
        self.vocab = spm.SentencePieceProcessor(model_file=directory + vocab_file)
        self.vocab_file = vocab_file # Save it for subsets

        self.sources = sources # Choose which section as input
        self.targets = targets # Choose which section as output
        self.max_views = max_views
        self.view_pos = view_pos
        self.max_len = max_len
        
        self.dir = directory
        self.input_size = input_size
        self.random_transform = random_transform
        self.__input_data(binary_mode=True)
        
        if random_transform:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.1,0.1,0.1), 
                    transforms.RandomRotation(15, expand=True)]),
                transforms.Resize(input_size),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([transforms.Resize(input_size), transforms.ToTensor()])
            
    def __len__(self):
        return len(self.idx_pidsid)

    def __getitem__(self, idx):
        idx = self.idx_pidsid[idx]

        sources = []
        targets = []

        # ------ Multiview Images ------
        if 'image' in self.sources:
            imgs, vpos = [], []
            
            # Randomly select V images from each folder 
            new_orders = np.random.permutation(len(self.img_files[idx]))
            img_files = np.array(self.img_files[idx])[new_orders].tolist()
            for i in range(min(self.max_views,len(img_files))):
                img_file = self.dir + 'images/' + idx[0] + '/' + idx[1] + '/' + img_files[i]
                pos = self.img_positions[img_files[i][:-4]]
                img = Image.open(img_file).convert('RGB')
                imgs.append(self.transform(img).unsqueeze(0)) # (1,C,W,H)
                vpos.append(self.dict_positions[pos])
            
            # If the number of images is smaller than V, pad the tensor with dummy images
            cur_len = len(vpos)
            for i in range(cur_len, self.max_views):
                imgs.append(torch.zeros_like(imgs[0]))
                vpos.append(-1) # Empty mask
            
            imgs = torch.cat(imgs, dim=0) # (V,C,W,H)
            vpos = np.array(vpos, dtype=np.int64) # (V)

        # ------ Additional Information ------
        info = self.img_captions[idx]
        
        source_info = []
        for section, content in info.items():
            if section in self.source_sections:
                source_info.append(content)
        source_info = ' '.join(source_info)
        
        encoded_source_info = [self.vocab.bos_id()] + self.vocab.encode(source_info) + [self.vocab.eos_id()]
        source_info = np.ones(self.max_len, dtype=np.int64) * self.vocab.pad_id()
        source_info[:min(len(encoded_source_info), self.max_len)] = encoded_source_info[:min(len(encoded_source_info), self.max_len)]

        target_info = []
        for section, content in info.items():
            if section in self.target_sections:
                target_info.append(content)
        target_info = ' '.join(target_info)
        
        # Compute extra labels (noun phrases)
        np_labels = np.zeros(len(self.top_np), dtype=float)
        for i in range(len(self.top_np)):
            if self.top_np[i] in target_info:
                np_labels[i] = 1
                
        encoded_target_info = [self.vocab.bos_id()] + self.vocab.encode(target_info) + [self.vocab.eos_id()]
        target_info = np.ones(self.max_len, dtype=np.int64) * self.vocab.pad_id()
        target_info[:min(len(encoded_target_info), self.max_len)] = encoded_target_info[:min(len(encoded_target_info), self.max_len)]

        for i in range(len(self.sources)):
            if self.sources[i] == 'image':
                sources.append((imgs,vpos))
            if self.sources[i] == 'history':
                sources.append(source_info)
            if self.sources[i] == 'label':
                sources.append(np.concatenate([self.img_labels[idx], np_labels]))
            if self.sources[i] == 'caption':
                sources.append(target_info)
            if self.sources[i] == 'caption_length':
                sources.append(min(len(encoded_target_info), self.max_len))
                
        for i in range(len(self.targets)):
            if self.targets[i] == 'label':
                targets.append(np.concatenate([self.img_labels[idx], np_labels]))
            if self.targets[i] == 'caption':
                targets.append(target_info)
            if self.targets[i] == 'caption_length':
                targets.append(min(len(encoded_target_info), self.max_len))
                
        return sources if len(sources) > 1 else sources[0], targets if len(targets) > 1 else targets[0]

    def __get_reports_images(self, file_name='reports.json'):
        caption_file = json.load(open(self.dir + file_name, 'r'))
        img_captions = {}
        img_files = {}
        for file_name, report in caption_file.items():
            k = file_name[-23:-4]
            pid,sid = k.split('/')
            try:
                # List all available images in each folder
                file_list = os.listdir(self.dir + 'images/' + pid + '/' + sid)
                # Select only images in self.view_pos
                file_list = [f for f in file_list if self.img_positions[f[:-4]] in self.view_pos]
                # Make sure there is atleast one image in each folder, and a non-empty findings section in each report
                if len(file_list) and ('FINDINGS:' in report) and (report['FINDINGS:'] != ''): 
                    img_files[(pid,sid)] = file_list
                    img_captions[(pid,sid)] = report
            except Exception as e:
                pass
        return img_captions, img_files

    def __get_view_positions(self, file_name='mimic-cxr-2.0.0-metadata.csv'):
        txt_file = self.dir + file_name
        data = pd.read_csv(txt_file, dtype=object)
        data = data.to_numpy().astype(str)
        return dict(zip(data[:,0].tolist(), data[:,4].tolist())), np.unique(data[:,4]).tolist()

    def __get_labels(self, binary_mode, file_name='mimic-cxr-2.0.0-chexpert.csv'):
        txt_file = self.dir + 'mimic-cxr-2.0.0-chexpert.csv'
        data = pd.read_csv(txt_file, dtype=object)

        label_names = list(data.columns.values[2:])
        data = data.to_numpy().astype(str)
        if binary_mode:
            data[data == '-1.0'] = "1" # 2 Not sure
            data[data ==  'nan'] = "0" # 3 Not mentioned
        else:
            data[data == '-1.0'] = "2" # 2 Not sure
            data[data ==  'nan'] = "3" # 3 Not mentioned
        
        img_labels = {}
        for i in range(len(data)):
            pid = 'p' + data[i,0].item()
            sid = 's' + data[i,1].item()
            labels = data[i,2:].astype(float)
            img_labels[(pid,sid)] = labels
        return img_labels, label_names

    def __get_nounphrase(self, top_k=100, file_name='count_nounphrase.json'):
        count_np = json.load(open(self.dir + file_name, 'r'))
        sorted_count_np = sorted([(k,v) for k,v in count_np.items()], key=lambda x: x[1], reverse=True)
        top_nounphrases = [k for k,v in sorted_count_np][:top_k]
        return top_nounphrases
           
    def __input_data(self, binary_mode=True):
        self.img_positions, self.list_positions = self.__get_view_positions()
        self.dict_positions = dict(zip(self.list_positions, range(len(self.list_positions))))
        self.img_captions, self.img_files = self.__get_reports_images()
        self.img_labels, self.list_diseases = self.__get_labels(binary_mode)
        self.dict_diseases = dict(zip(self.list_diseases, range(len(self.list_diseases))))
        self.idx_pidsid = list(self.img_captions.keys())
        self.top_np = self.__get_nounphrase()
        
    def __generate_splits(self, test_size=0.2, seed=0, file_name='mimic-cxr-2.0.0-chexpert.csv'):
        train_val_file = open(self.dir + 'train_val_list.txt', 'w')
        test_file = open(self.dir + 'test_list.txt', 'w')

        txt_file = self.dir + 'mimic-cxr-2.0.0-chexpert.csv'
        data = pd.read_csv(txt_file, dtype=object)
        data = data.to_numpy().astype(str)

        # 1 PID can have multiple SIDs
        pid_sid = {}
        for i in range(len(data)):
            pid = data[i,0].item()
            sid = data[i,1].item() 
            
            if pid in pid_sid:
                pid_sid[pid].append(sid)
            else:
                pid_sid[pid] = [sid]

        np.random.seed(seed)
        unique_pid = np.unique(data[:,0])        
        random_pid = np.random.permutation(unique_pid)

        pvt = int((1-test_size) * len(unique_pid))
        train_pid = random_pid[:pvt]
        test_pid = random_pid[pvt:]

        for pid in train_pid:
            for sid in pid_sid[pid]:
                if ('p'+pid,'s'+sid) in self.img_captions:
                    train_val_file.write('p' + pid + '/' + 's' + sid + '\n')
        
        for pid in test_pid:
            for sid in pid_sid[pid]:
                if ('p'+pid,'s'+sid) in self.img_captions:
                    test_file.write('p' + pid + '/' + 's' + sid + '\n')

    def get_subsets(self, pvt=0.9, seed=0, generate_splits=True, debug_mode=False, train_phase=True):
        if generate_splits:
            self.__generate_splits(seed=0)
            print('New splits generated')
            
        train_files = np.loadtxt(self.dir + 'train_val_list.txt', dtype=str)
        test_files = np.loadtxt(self.dir + 'test_list.txt', dtype=str)

        train_files = np.array([f.split('/') for f in train_files])
        test_files = np.array([f.split('/') for f in test_files])
        
        np.random.seed(seed)
        indices = np.random.permutation(len(train_files))
        pivot = int(len(train_files) * pvt)
        train_indices = indices[:pivot]
        val_indices = indices[pivot:]

        train_dataset = MIMIC(self.dir, self.input_size, self.random_transform, 
                              self.view_pos, self.max_views, self.sources, self.targets, 
                              self.max_len, self.vocab_file)
        train_dataset.idx_pidsid = [(pid,sid) for pid,sid in train_files[train_indices]] if not debug_mode else [(pid,sid) for pid,sid in train_files[train_indices]][:10000]
        
        val_dataset = MIMIC(self.dir, self.input_size, False, 
                            self.view_pos, self.max_views, self.sources, self.targets, 
                            self.max_len, self.vocab_file)
        val_dataset.idx_pidsid = [(pid,sid) for pid,sid in train_files[val_indices]] if not debug_mode else [(pid,sid) for pid,sid in train_files[val_indices]][:1000]

        test_dataset = MIMIC(self.dir, self.input_size, False, 
                            self.view_pos, self.max_views, self.sources, self.targets, 
                            self.max_len, self.vocab_file)
        test_dataset.idx_pidsid = [(pid,sid) for pid,sid in test_files] if not debug_mode else [(pid,sid) for pid,sid in test_files][:1000]

        # Use only a subset to make the model run quickly
        if train_phase:
            subset_size = 1000
        else:
            subset_size = 100#000
        
        val_idx = np.random.choice(len(val_dataset.idx_pidsid), size=min(subset_size, len(val_dataset.idx_pidsid)), replace=False)
        test_idx = np.random.choice(len(test_dataset.idx_pidsid), size=min(subset_size, len(test_dataset.idx_pidsid)), replace=False)
        
        train_dataset.idx_pidsid = train_dataset.idx_pidsid[:]
        val_dataset.idx_pidsid = [val_dataset.idx_pidsid[i] for i in val_idx]
        test_dataset.idx_pidsid = [test_dataset.idx_pidsid[i] for i in test_idx]
        
        return train_dataset, val_dataset, test_dataset

class NLMCXR(data.Dataset): # Open-I Dataset
    def __init__(self, directory, IMAGE_MODE, input_size=(256,256), random_transform=True,
                view_pos=['AP', 'PA', 'LATERAL'], max_views=2, sources=['image','history'], targets=['label'], 
                max_len=1000, vocab_file='Vocabulary/nlmcxr_unigram_1000.model'):
        
        self.source_sections = ['INDICATION', 'COMPARISON']
        self.target_sections = ['FINDINGS']
        self.vocab = spm.SentencePieceProcessor(model_file=vocab_file)
        self.vocab_file = vocab_file # Save it for subsets

        self.sources = sources # Choose which section as input
        self.targets = targets # Choose which section as output
        self.max_views = max_views
        self.view_pos = view_pos
        self.max_len = max_len

        self.dir = directory
        self.input_size = input_size
        self.random_transform = random_transform
        self.__input_data(binary_mode=True)
        self.image_mode = IMAGE_MODE
        self.ent_num=1043
        
        
        if random_transform:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.1,0.1,0.1), 
                    transforms.RandomRotation(15, expand=True)]),
                transforms.Resize(input_size),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([transforms.Resize(input_size), transforms.ToTensor()])
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        #TODO:file random mask with zero, patchify?
        file_name = self.file_list[idx]
        sources, targets = [], []
        tmp_rep = self.captions[self.file_report[file_name]['image'][0] + '.png']
        
        # ------ Multiview Images ------
        #TODO:without multiview images
        if 'image' in self.sources:
            imgs, vpos = [], []
            images = self.file_report[file_name]['image']

            # Randomly select V images from each folder 
            new_orders = np.random.permutation(len(images))
            img_files = np.array(images)[new_orders].tolist()

            if self.image_mode == 'mask':
                for i in range(min(self.max_views,len(img_files))):
                    img_file = self.dir + 'images/' + img_files[i] + '.png'
                    # mask the random 50*50 area in images
                    img = cv2.imread(img_file,cv2.IMREAD_GRAYSCALE)
                    #print(img.shape)
                    if len(img.shape)==2:
                        height,width = img.shape
                        x = np.random.randint(0, width - 300)
                        y = np.random.randint(0, height - 300)
                        size = np.random.randint(0, 300)
                        image = img.copy()
                        y1 = y + size
                        x1 = x + size
                        #print(x,y,x1,y1)
                        #print(image[y:y1,x:x1])
                        img[y:y1,x:x1]=0
                        # img[y:y1, x:x1] = cv2.medianBlur(img[y:y1, x:x1], 133)  # 此为中值模糊，常用于去除椒盐噪声
                    elif len(img.shape)==3:
                        height,width,_ = img.shape
                        x = np.random.randint(0, width - 300)
                        y = np.random.randint(0, height- 300)
                        size = np.random.randint(0,300)
                        image = img.copy()
                        y1 = y+size
                        x1 = x+size
                        img[y:y1,x:x1]=0
                        # img[y:y1, x:x1,:] = cv2.medianBlur(img[y:y1, x:x1,:], 133)  # 此为中值模糊，常用于去除椒盐噪声
                    img = Image.fromarray(img).convert('RGB')
                imgs.append(self.transform(img).unsqueeze(0)) # (1,C,W,H)
                vpos.append(1) # We do not know what view position of the image is, so just let it be 1
                # If the number of images is smaller than V, pad the tensor with dummy images
                cur_len = len(vpos)
                for i in range(cur_len, self.max_views):
                    imgs.append(torch.zeros_like(imgs[0]))
                    vpos.append(-1) # Empty mask
                
                imgs = torch.cat(imgs, dim=0) # (V,C,W,H)
                vpos = np.array(vpos, dtype=np.int64) # (V)

                # TODO: visualize the images in tensorboard
            # elif self.image_mode == 'unbias':
            #     for i in range(len(img_files)):
            #         image_name = img_files[i]
            #         original_path = os.path.join('nlmcxr_feature_origin_new/', image_name+'.npy')
            #         if not os.path.exists(original_path):
            #             continue
            #         else:
            #             original_feature = np.load('nlmcxr_feature_origin_new/' + image_name+'.npy', allow_pickle=True)
            #     # if not original_feature.all():
            #         # print('original missing',img_files)
            #         original_feature = original_feature[np.newaxis, :,:]
            #     for i in range(len(img_files)):
            #         image_name = img_files[i]
            #         mask_path = os.path.join('nlmcxr_feature_mask_new/', image_name+'.npy')
            #         if not os.path.exists(mask_path):
            #             continue
            #         else:
            #             mask_features = np.load('nlmcxr_feature_mask_new/' + image_name+'.npy',allow_pickle=True)
            #         # if not mask_features.all():
            #             # print('mask missing: ',img_files)
            #             # mask_features = mask_features[np.newaxis,:,:]
            #         # else:
            #         mask_features = mask_features[np.newaxis, :,:]
            #     # print(original_feature.shape,mask_features.shape)
            #     if original_feature.shape != mask_features.shape:
            #         print(original_feature.shape,mask_features.shape)
            #     assert original_feature.shape == mask_features.shape
            #     imgs.append(original_feature[0])
            #     imgs.append(mask_features[0])
            else:
                for i in range(min(self.max_views,len(img_files))):
                    img_file = self.dir + 'images/' + img_files[i] + '.png'
                    img = Image.open(img_file).convert('RGB')
                imgs.append(self.transform(img).unsqueeze(0)) # (1,C,W,H)
                vpos.append(1) # We do not know what view position of the image is, so just let it be 1
                # If the number of images is smaller than V, pad the tensor with dummy images
                cur_len = len(vpos)
                for i in range(cur_len, self.max_views):
                    imgs.append(torch.zeros_like(imgs[0]))
                    vpos.append(-1) # Empty mask
                
                imgs = torch.cat(imgs, dim=0) # (V,C,W,H)
                vpos = np.array(vpos, dtype=np.int64) # (V)

        # ------ Additional Information ------
        info = self.file_report[file_name]['report']
        
        source_info = []
        for section, content in info.items():
            if section in self.source_sections:
                source_info.append(content)
        source_info = ' '.join(source_info)
        
        encoded_source_info = [self.vocab.bos_id()] + self.vocab.encode(source_info) + [self.vocab.eos_id()]
        source_info = np.ones(self.max_len, dtype=np.int64) * self.vocab.pad_id()
        source_info[:min(len(encoded_source_info), self.max_len)] = encoded_source_info[:min(len(encoded_source_info), self.max_len)]

        target_info = []
        for section, content in info.items():
            if section in self.target_sections:
                target_info.append(content)
        # target_info = ' '.join(target_info)
        target_info = tmp_rep # This load the document from our previous AAAI paper (preprocessed documents)
        
        # read entities
        if file_name in self.ent_list:
            entities = self.ent_list[file_name]
            entities = np.array(entities)
            np_ents = np.zeros(self.ent_num, dtype=float)
            for i in range(self.ent_num):
                if i in entities:
                    np_ents[i] = 1
        else:
            np_ents = np.zeros(self.ent_num, dtype=float)


        np_labels = np.zeros(len(self.top_np), dtype=float)
        for i in range(len(self.top_np)):
            if self.top_np[i] in target_info:
                np_labels[i] = 1
        
        encoded_target_info = [self.vocab.bos_id()] + self.vocab.encode(target_info) + [self.vocab.eos_id()]
        target_info = np.ones(self.max_len, dtype=np.int64) * self.vocab.pad_id()
        target_info[:min(len(encoded_target_info), self.max_len)] = encoded_target_info[:min(len(encoded_target_info), self.max_len)]

        for i in range(len(self.sources)):
            if self.sources[i] == 'image':
            #     if self.image_mode == 'unbias':
            #         sources.append(imgs)
            #     else:
                sources.append((imgs,vpos))
            if self.sources[i] == 'entity':
                sources.append(np_ents)
            if self.sources[i] == 'history':
                sources.append(source_info)
            if self.sources[i] == 'label':
                sources.append(np.concatenate([np.array(self.file_labels[file_name]), np_labels]))
            if self.sources[i] == 'caption':
                sources.append(target_info)
            if self.sources[i] == 'caption_length':
                sources.append(min(len(encoded_target_info), self.max_len))
            if self.sources[i] == 'image_name':
                sources.append(self.file_report[file_name]['image'])
                
        for i in range(len(self.targets)):
            if self.targets[i] == 'entity':
                targets.append(np_ents)
            if self.targets[i] == 'label':
                targets.append(np.concatenate([np.array(self.file_labels[file_name]), np_labels]))
            if self.targets[i] == 'caption':
                targets.append(target_info)
            if self.targets[i] == 'caption_length':
                targets.append(min(len(encoded_target_info), self.max_len))
        # print(len(sources))
        # print(len(targets))
        return sources if len(sources) > 1 else sources[0], targets if len(targets) > 1 else targets[0]

    def __get_nounphrase(self, top_k=100, file_name='count_nounphrase.json'):
        count_np = json.load(open(os.path.join('open-i',file_name), 'r'))
        sorted_count_np = sorted([(k,v) for k,v in count_np.items()], key=lambda x: x[1], reverse=True)
        top_nounphrases = [k for k,v in sorted_count_np][:top_k]
        return top_nounphrases

    def __input_data(self, binary_mode=True):
        self.__input_caption()
        self.__input_report()
        self.__input_ent()
        self.__input_label()
        self.__filter_inputs()
        self.top_np = self.__get_nounphrase()
    
    # Input entities
    def __input_ent(self):
        self.ent_list = {}
        self.ent_short_list = []
        with open('data/pretrain_data/cache_data.pkl','rb') as j:
            data = pkl.load(j)

        self.ent2id = open("data/knowledge/entity2id.txt").read().strip().split('\n')[1:]
        self.ent2id = {kv.split("\t")[0]: kv.split('\t')[1] for kv in self.ent2id}

        for k,v in data.items():
            for i,j in v.items():
                for x in j:
                    #print('----------ent x-----------',x)
                    self.ent_list[x['img_path']] = [int(self.ent2id[a[-1]]) for a in x['image_entities'] if a[-1] in self.ent2id]
                    self.ent_short_list.append('/'.join(x['img_path'].split('/')[10:12]))
        #with open('entlist.json','w') as j:
        #    json.dump(self.ent_list,j)

    def __input_label(self):
        with open('open-i/file2label.json') as f:
            labels = json.load(f)
        self.file_labels = labels
        
    def __input_caption(self):
        with open('open-i/captions.json') as f:
            captions = json.load(f)
        self.captions = captions
        
    def __input_report(self):
        with open('open-i/reports_ori.json') as f:
            reports = json.load(f)
        self.file_list = [k for k in reports.keys()]
        self.file_report = reports

    def __filter_inputs(self):
        filtered_file_report = {}
        for k, v in self.file_report.items():
            if (len(v['image']) > 0) and (('FINDINGS' in v['report']) and (v['report']['FINDINGS'] != '')): # or (('IMPRESSION' in v['report']) and (v['report']['IMPRESSION'] != ''))):
                filtered_file_report[k] = v
        self.file_report = filtered_file_report
        self.file_list = [k for k in self.file_report.keys()]

    def get_subsets(self, IMAGE_MODE, train_size=0.7, val_size=0.1, test_size=0.2, seed=0):
        # read train,val,test from json file
        with open('new_nlmcxr_datasets.json','r') as f:
            all_data = json.load(f)
        train_list = all_data['train']
        val_list = all_data['val']
        test_list = all_data['test']

        train_dataset = NLMCXR(self.dir, IMAGE_MODE, self.input_size, self.random_transform, 
                              self.view_pos, self.max_views, self.sources, self.targets, self.max_len, self.vocab_file)
        train_dataset.file_list = train_list

        # Consider change random_transform to False for validation
        val_dataset = NLMCXR(self.dir, IMAGE_MODE, self.input_size, False, 
                            self.view_pos, self.max_views, self.sources, self.targets, self.max_len, self.vocab_file)
        val_dataset.file_list = val_list

        # Consider change random_transform to False for testing
        test_dataset = NLMCXR(self.dir, IMAGE_MODE, self.input_size, False, 
                             self.view_pos, self.max_views, self.sources, self.targets, self.max_len, self.vocab_file)
        test_dataset.file_list = test_list

        all_dataset = NLMCXR(self.dir, IMAGE_MODE, self.input_size, False, 
                             self.view_pos, self.max_views, self.sources, self.targets, self.max_len, self.vocab_file)
        all_dataset.file_list = train_list+val_list+test_list
        return all_dataset, train_dataset, val_dataset, test_dataset
    
class TextDataset(data.Dataset):
    def __init__(self, text_file, label_file, sources=['caption'], targets=['label'],
                 vocab_file='/home/hoang/Datasets/MIMIC/mimic_unigram_1000.model', max_len=1000):
        self.text_file = text_file
        self.label_file = label_file
        self.vocab = spm.SentencePieceProcessor(model_file=vocab_file)
        self.sources = sources # Choose which section as input
        self.targets = targets # Choose which section as output
        self.max_len = max_len
        self.__input_data()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        encoded_text = [self.vocab.bos_id()] + self.vocab.encode(self.lines[idx].strip()) + [self.vocab.eos_id()]
        text = np.ones(self.max_len, dtype=np.int64) * self.vocab.pad_id()
        text[:min(len(encoded_text), self.max_len)] = encoded_text[:min(len(encoded_text), self.max_len)]
        
        sources = []
        for i in range(len(self.sources)):
            if self.sources[i] == 'label':
                sources.append(self.labels[idx])
            if self.sources[i] == 'caption':
                sources.append(text)
            if self.sources[i] == 'caption_length':
                sources.append(min(len(encoded_text), self.max_len))
        
        targets = []
        for i in range(len(self.targets)):
            if self.targets[i] == 'label':
                targets.append(self.labels[idx])
            if self.targets[i] == 'caption':
                targets.append(text)
            if self.targets[i] == 'caption_length':
                targets.append(min(len(encoded_text), self.max_len))
                
        return sources if len(sources) > 1 else sources[0], targets if len(targets) > 1 else targets[0]
    
    def __input_data(self):
        data_file = open(self.text_file, 'r') 
        self.lines = data_file.readlines()
        self.labels = np.loadtxt(self.label_file, dtype='float')


if __name__ == "__main__":
    DATASET_PATH = '/home/tianyan/datasets/NLMCXR/'
    SOURCES = ['image','caption','label','history']
    TARGETS = ['caption','label']
    dataset = NLMCXR(DATASET_PATH, IMAGE_MODE='train', view_pos=['AP','PA','LATERAL'], max_views=2, sources=SOURCES, targets=TARGETS)
    train_data, val_data, test_data = dataset.get_subsets(IMAGE_MODE='train', seed=123)
