import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from data.preprocessing import DIAGNOSTIC_MAP

class PADUFESDataset(Dataset):
    """
    Multimodal PyTorch Dataset for PAD-UFES-20.
    Supports Image (ViT), Metadata (MLP), and Free Text (ClinicalBERT).
    """
    def __init__(self, df, img_dir, meta_features, text_list, tokenizer=None,
                 img_transform=None, max_seq_length=128):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.meta_features = meta_features
        self.text_list = text_list
        self.tokenizer = tokenizer
        self.img_transform = img_transform
        self.max_seq_length = max_seq_length

        self.labels = [DIAGNOSTIC_MAP[diag] for diag in self.df['diagnostic']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row['img_id'])
        img_path = os.path.join(self.img_dir, img_id)

        # 1. Image loading
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception:
                img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        else:
            img = Image.new('RGB', (224, 224), color=(128, 128, 128))

        if self.img_transform is not None:
            img_tensor = self.img_transform(img)
        else:
            img_tensor = torch.zeros(3, 224, 224)

        # 2. Metadata loading
        meta_tensor = torch.tensor(self.meta_features[idx], dtype=torch.float32)

        # 3. Text Tokenization
        text = self.text_list[idx]
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors='pt'
            )
            input_ids = tokens['input_ids'].squeeze(0)
            attention_mask = tokens['attention_mask'].squeeze(0)
        else:
            input_ids = torch.zeros(self.max_seq_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_seq_length, dtype=torch.long)

        # 4. Target Label
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # 5. Missingness fraction
        missing_frac = float(row.isnull().mean()) if hasattr(row, 'isnull') else 0.0

        return {
            'image': img_tensor,
            'metadata': meta_tensor,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label,
            'img_id': img_id,
            'text': text,
            'missing_fraction': missing_frac
        }
