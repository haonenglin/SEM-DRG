import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class BaseDataset(Dataset):

    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.entity_dir = args.entity_dir
        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])
        self.entity_features = self._load_all_entity_features() if getattr(args, 'use_entity_features', True) else {}

    def _load_all_entity_features(self):
        entity_dir = self.entity_dir
        feature_names = ['complication_features', 'source_features', 'tube_features', 'volume_features', 'clarity_features', 'color_features', 'wall_attachment_features', 'precipitate_features', 'precipitate_color_features']
        features_dict = {}
        for name in feature_names:
            npy_path = os.path.join(entity_dir, f'{name}.npy')
            if os.path.exists(npy_path):
                data = np.load(npy_path)
                features_dict[name] = torch.tensor(data, dtype=torch.float32)
            else:
                raise FileNotFoundError(npy_path)
        return features_dict

    def _get_entity_features_for_sample(self, idx):
        ent_lists = []
        if self.entity_features:
            for name in ['complication_features', 'source_features', 'tube_features', 'volume_features', 'clarity_features', 'color_features', 'wall_attachment_features', 'precipitate_features', 'precipitate_color_features']:
                features = self.entity_features[name]
                if features is not None:
                    if idx < len(features):
                        ent_lists.append(features[idx])
                    else:
                        ent_lists.append(features[-1])
                else:
                    ent_lists.append(None)
        return ent_lists

    def __len__(self):
        return len(self.examples)

class DrainageFluidBag_Entity_Dataset(BaseDataset):

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        ent_lists = self._get_entity_features_for_sample(idx)
        sample = (image_id, image, report_ids, report_masks, seq_length, ent_lists)
        return sample
