import numpy as np
import pandas as pd
import random as random
from torch.utils.data import Dataset
import os
from PIL import Image

class NIHDataset(Dataset):

    def __init__(
            self,
            path_to_nih_folder,
            fold,
            transform=None,
            transform_bbox=None):
        
        self.transform = transform
        self.transform_bbox = transform_bbox
        self.path_to_nih_folder = path_to_nih_folder
        self.fold = fold
        self.df = pd.read_csv(os.path.join(path_to_nih_folder, "labels", "nih_original_split.csv"))
        
        if not fold == 'bbox':
            self.df = self.df[self.df['fold'] == fold]
        else:
            bbox_images_df = pd.read_csv(os.path.join(path_to_nih_folder, "labels", "BBox_List_2017.csv"))
            self.df=pd.merge(left=self.df, right=bbox_images_df, how="inner", on="Image Index")
            self.df = self.df.replace('Infiltrate', 'Infiltration')

                
        self.df = self.df.set_index("Image Index")
        self.PRED_LABEL = [
            'Atelectasis',
            'Cardiomegaly',
            'Effusion',
            'Infiltration',
            'Mass',
            'Nodule',
            'Pneumonia',
            'Pneumothorax',
            'Consolidation',
            'Edema',
            'Emphysema',
            'Fibrosis',
            'Pleural_Thickening',
            'Hernia']

                

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image = Image.open(
            os.path.join(
                self.path_to_nih_folder, 
                "images",
                self.df.index[idx]))
        image = image.convert('RGB')

        label = np.zeros(len(self.PRED_LABEL), dtype=int)
        for i in range(0, len(self.PRED_LABEL)):
             # can leave zero if zero, else make one
            if(self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') in set([-1,1])):
                label[i] = self.df[self.PRED_LABEL[i].strip()
                                   ].iloc[idx].astype('int')

        if self.transform:
            image = self.transform(image)
            
        if self.fold == "bbox":
            # exctract bounding box coordinates from dataframe, they exist in the the columns specified below
            #bounding_box = self.df.iloc[idx, -7:-3].as_matrix()
            bounding_box = self.df.iloc[idx, -7:-3]
            bounding_box_label = self.df.iloc[idx, -8]

            if self.transform_bbox:
                transformed_bounding_box = self.transform_bbox(bounding_box)

            return image, label, self.df.index[idx], transformed_bounding_box, bounding_box_label
        else:
            return image, label, self.df.index[idx]
    
        
    def pos_neg_sample_nums(self):
        df = self.df[self.PRED_LABEL]
        N, C = df.shape
        pos_neg_sample_nums = np.zeros((2,C))
        pos_neg_sample_nums[0, :] = df.sum(axis=0).values
        pos_neg_sample_nums[1, :] = N - pos_neg_sample_nums[0, :]
        del df
        return pos_neg_sample_nums
        

