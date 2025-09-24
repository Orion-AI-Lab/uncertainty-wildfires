from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import pandas as pd
import random
import warnings

class FireDataset(Dataset):
    def __init__(self, dataset_root: Path = None, problem_class: str = 'classification', train_val_test: str = 'train', dynamic_features: list = None,
                 static_features: list = None, nan_fill: float = 0.0, lag: int = 45, temporal_gap = 0, neg_pos_ratio_train: int = 2, neg_pos_ratio_test: int = 2, seed: int = 12345):
            
        self.static_features = static_features
        self.dynamic_features = dynamic_features
        self.problem_class = problem_class
        self.nan_fill = nan_fill
        self.lag = lag
        self.temporal_gap = temporal_gap
        self.seed = seed
        self.x = None
        self.y = None
     
        random.seed(self.seed)

        assert problem_class in ['classification', 'segmentation']

        dataset_root = Path(dataset_root)
        
        self.positives = pd.read_csv(dataset_root / 'positives.csv')
        self.positives['label'] = 1
        self.negatives = pd.read_csv(dataset_root / 'negatives.csv')
        self.negatives['label'] = 0
        
        def get_last_year(group):
            last_date = group['time'].iloc[-1]
            return last_date[:4]

        self.positives['YEAR'] = self.positives.groupby(self.positives.index // 60).apply(get_last_year).values.repeat(60)
        self.negatives['YEAR'] = self.negatives.groupby(self.negatives.index // 60).apply(get_last_year).values.repeat(60)   
        
        val_year = ['2020']
        test_year = ['2021', '2022']

        self.train_positives = self.positives[~self.positives['YEAR'].isin(val_year + test_year)]
        self.val_positives = self.positives[self.positives['YEAR'].isin(val_year)]
        self.test_positives = self.positives[self.positives['YEAR'].isin(test_year)]
        
        bas_median = self.train_positives['burned_area_has'].median()
        
        def random_(negatives, positives, neg_pos_ratio):
            ids_selected = np.random.choice(negatives['sample'].unique(), (len(positives)//60)*neg_pos_ratio, replace=False)
            negatives = negatives[negatives['sample'].isin(ids_selected)]
            return negatives
        
        self.train_negatives = random_(self.negatives[~self.negatives['YEAR'].isin(val_year + test_year)], self.train_positives, neg_pos_ratio_train)
        self.val_negatives = random_(self.negatives[self.negatives['YEAR'].isin(val_year)], self.val_positives, neg_pos_ratio_test)
        self.test_negatives = random_(self.negatives[self.negatives['YEAR'].isin(test_year)], self.test_positives, neg_pos_ratio_test)


        if train_val_test == 'train':
            print(f'Positives: {len(self.train_positives)/60} / Negatives: {len(self.train_negatives)/60}')
            self.all = pd.concat([self.train_positives, self.train_negatives]).reset_index()
        elif train_val_test == 'val':
            print(f'Positives: {len(self.val_positives)/60} / Negatives: {len(self.val_negatives)/60}')
            self.all = pd.concat([self.val_positives, self.val_negatives]).reset_index()
        elif train_val_test == 'test':
            print(f'Positives: {len(self.test_positives)/60} / Negatives: {len(self.test_negatives)/60}')
            self.all = pd.concat([self.test_positives, self.test_negatives]).reset_index()
                       
        print("Dataset length", len(self.all)/60)
                       
        self.labels = self.all.label.tolist()
        self.dynamic = self.all[self.dynamic_features]
        self.static = self.all[self.static_features]
        self.x = self.all['x']
        self.y = self.all['y']
   
                       
        self.dynamic = (self.dynamic - self.dynamic.mean()) / self.dynamic.std()
        self.static = (self.static - self.static.mean()) / self.static.std()
        
        self.burned_areas_size = self.all['burned_area_has'].replace(0, 30)

        
                       
    def __len__(self):
        return int(len(self.all)/60)

    def __getitem__(self, idx):
        dynamic = self.dynamic.iloc[idx*60:(idx+1)*60].values[-self.lag - self.temporal_gap:60-self.temporal_gap, :]
        static = self.static.iloc[idx*60:(idx+1)*60].values[0,:]
        burned_areas_size = np.log(self.burned_areas_size.iloc[idx*60:(idx+1)*60].values[0])
        labels = self.labels[idx*60]        
        x = self.x.iloc[idx*60]
        y = self.y.iloc[idx*60]


        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            feat_mean = np.nanmean(dynamic, axis=0)
            # Find indices that you need to replace
            inds = np.where(np.isnan(dynamic))
            # Place column means in the indices. Align the arrays using take
            dynamic[inds] = np.take(feat_mean, inds[1])

        dynamic = np.nan_to_num(dynamic, nan=self.nan_fill)
        static = np.nan_to_num(static, nan=self.nan_fill)

        return dynamic, static, burned_areas_size, labels, x, y
