import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import torchvision.transforms as T

DIAGNOSTIC_MAP = {
    'BCC': 0,
    'ACK': 1,
    'NEV': 2,
    'SEK': 3,
    'SCC': 4,
    'MEL': 5
}

DIAGNOSTIC_NAMES = ['BCC', 'ACK', 'NEV', 'SEK', 'SCC', 'MEL']

CATEGORICAL_COLS = [
    'smoke', 'drink', 'background_father', 'background_mother',
    'pesticide', 'gender', 'skin_cancer_history', 'cancer_history',
    'has_piped_water', 'has_sewage_system', 'fitspatrick', 'region',
    'itch', 'grew', 'hurt', 'changed', 'bleed', 'elevation', 'biopsed'
]

NUMERICAL_COLS = ['age', 'diameter_1', 'diameter_2']

# High sun-exposure regions
SUN_EXPOSED_REGIONS = {'FACE', 'NOSE', 'EAR', 'NECK', 'SCALP', 'LIP', 'FOREARM', 'HAND', 'ARM'}

class MetadataPreprocessor:
    """
    Advanced Metadata Preprocessor with Domain-Specific Dermatological Feature Engineering:
    - Diameter aspect ratio (elongation index)
    - Estimated lesion area (elliptical approximation)
    - High-risk symptom severity score (bleed, hurt, grew, changed, elevation, itch)
    - Sun-exposure anatomical risk indicator
    - Fitzpatrick skin vulnerability indexing
    - Missingness indicator flags (preserving informative clinical missingness)
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.is_fitted = False

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 1. Diameter aspect ratio and area
        d1 = pd.to_numeric(df['diameter_1'], errors='coerce').fillna(0.0) if 'diameter_1' in df.columns else pd.Series(0.0, index=df.index)
        d2 = pd.to_numeric(df['diameter_2'], errors='coerce').fillna(0.0) if 'diameter_2' in df.columns else pd.Series(0.0, index=df.index)
        df['diameter_1'] = d1
        df['diameter_2'] = d2
        df['diameter_ratio'] = np.where(d2 > 0, d1 / (d2 + 1e-5), 1.0)
        df['lesion_area'] = np.pi * (d1 / 2.0) * (d2 / 2.0)
        df['max_diameter'] = np.maximum(d1, d2)

        # 2. High-risk symptom cumulative score
        symptom_cols = ['bleed', 'hurt', 'grew', 'changed', 'elevation', 'itch']
        symptom_score = np.zeros(len(df))
        for col in symptom_cols:
            if col in df.columns:
                is_true = (df[col].astype(str).str.upper() == 'TRUE').astype(float)
                symptom_score += is_true
            else:
                df[col] = 'Unknown'
        df['symptom_severity_score'] = symptom_score

        # 3. Sun exposure risk
        if 'region' in df.columns:
            region_upper = df['region'].astype(str).str.upper()
        else:
            region_upper = pd.Series('UNKNOWN', index=df.index)
            df['region'] = 'Unknown'
        df['is_sun_exposed'] = region_upper.isin(SUN_EXPOSED_REGIONS).astype(float)

        # 4. Age risk groups (<30, 30-50, 50-70, >70)
        if 'age' in df.columns:
            age = pd.to_numeric(df['age'], errors='coerce').fillna(50.0)
        else:
            age = pd.Series(50.0, index=df.index)
            df['age'] = 50.0
        df['age_under_30'] = (age < 30).astype(float)
        df['age_30_50'] = ((age >= 30) & (age < 50)).astype(float)
        df['age_50_70'] = ((age >= 50) & (age < 70)).astype(float)
        df['age_over_70'] = (age >= 70).astype(float)

        # 5. Missingness fraction per record (for Q_meta)
        df['missing_fraction'] = df.isnull().mean(axis=1)

        # 6. Dermatological Diagnostic Likelihood Scores
        is_face = region_upper.isin({'FACE', 'NOSE', 'EAR', 'LIP', 'NECK'}).astype(float)
        is_extremity = region_upper.isin({'ARM', 'FOREARM', 'HAND', 'FOOT', 'THIGH'}).astype(float)
        is_trunk = region_upper.isin({'CHEST', 'BACK', 'ABDOMEN'}).astype(float)

        bleed_v = (df['bleed'].astype(str).str.upper() == 'TRUE').astype(float) if 'bleed' in df.columns else 0.0
        elev_v = (df['elevation'].astype(str).str.upper() == 'TRUE').astype(float) if 'elevation' in df.columns else 0.0
        hurt_v = (df['hurt'].astype(str).str.upper() == 'TRUE').astype(float) if 'hurt' in df.columns else 0.0
        changed_v = (df['changed'].astype(str).str.upper() == 'TRUE').astype(float) if 'changed' in df.columns else 0.0
        grew_v = (df['grew'].astype(str).str.upper() == 'TRUE').astype(float) if 'grew' in df.columns else 0.0
        itch_v = (df['itch'].astype(str).str.upper() == 'TRUE').astype(float) if 'itch' in df.columns else 0.0

        max_d_v = df['max_diameter'].values
        ratio_v = df['diameter_ratio'].values
        age_norm = np.clip(age.values / 70.0, 0.0, 1.0)

        df['bcc_score'] = 0.4 * is_face.values + 0.3 * (bleed_v + elev_v) / 2.0 + 0.3 * age_norm
        df['ack_score'] = 0.4 * np.maximum(is_face.values, is_extremity.values) + 0.3 * itch_v + 0.3 * (1.0 - elev_v)
        df['nev_score'] = 0.4 * (1.0 - np.clip(age.values / 45.0, 0.0, 1.0)) + 0.3 * is_trunk.values + 0.3 * (1.0 - bleed_v) * (1.0 - hurt_v)
        df['sek_score'] = 0.3 * age_norm + 0.3 * elev_v + 0.4 * is_trunk.values
        df['scc_score'] = 0.3 * is_face.values + 0.3 * (hurt_v + grew_v + bleed_v) / 3.0 + 0.4 * np.clip(max_d_v / 15.0, 0.0, 1.0)
        df['mel_score'] = 0.3 * (changed_v + grew_v) / 2.0 + 0.3 * (max_d_v > 6.0).astype(float) + 0.4 * np.maximum(is_trunk.values, is_extremity.values)
        df['abcde_score'] = (ratio_v > 1.3).astype(float) + changed_v + (max_d_v > 6.0).astype(float) + ((grew_v + bleed_v + elev_v) > 0).astype(float)

        # 7. Non-linear Interaction & Synergistic Risk Features
        df['log_lesion_area'] = np.log1p(df['lesion_area'].values)
        df['bleed_grew_interaction'] = bleed_v * grew_v
        df['hurt_bleed_interaction'] = hurt_v * bleed_v
        df['sun_age_interaction'] = df['is_sun_exposed'].values * age_norm
        df['risk_ratio'] = np.where(max_d_v > 0, age.values / (max_d_v + 1e-4), 0.0)
        df['melanoma_high_risk_flag'] = ((changed_v == 1.0) & (max_d_v > 6.0) & (ratio_v > 1.2)).astype(float)
        df['carcinoma_high_risk_flag'] = (((bleed_v == 1.0) | (hurt_v == 1.0)) & (age_norm > 0.6)).astype(float)

        return df

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        df_feat = self._engineer_features(df)
        
        # Fill missing categoricals
        for col in CATEGORICAL_COLS:
            if col in df_feat.columns:
                df_feat[col] = df_feat[col].astype(str).replace({'nan': 'Unknown', 'UNK': 'Unknown', 'None': 'Unknown'})
            else:
                df_feat[col] = 'Unknown'

        # Impute and scale numeric features
        engineered_num_cols = NUMERICAL_COLS + [
            'diameter_ratio', 'lesion_area', 'max_diameter',
            'symptom_severity_score', 'is_sun_exposed',
            'age_under_30', 'age_30_50', 'age_50_70', 'age_over_70', 'missing_fraction',
            'bcc_score', 'ack_score', 'nev_score', 'sek_score', 'scc_score', 'mel_score', 'abcde_score',
            'log_lesion_area', 'bleed_grew_interaction', 'hurt_bleed_interaction',
            'sun_age_interaction', 'risk_ratio', 'melanoma_high_risk_flag', 'carcinoma_high_risk_flag'
        ]
        num_data = self.num_imputer.fit_transform(df_feat[engineered_num_cols])
        num_scaled = self.scaler.fit_transform(num_data)

        # One-hot encode categoricals
        cat_data = self.cat_imputer.fit_transform(df_feat[CATEGORICAL_COLS])
        cat_encoded = self.ohe.fit_transform(cat_data)

        self.is_fitted = True
        return np.hstack([num_scaled, cat_encoded]).astype(np.float32)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("MetadataPreprocessor must be fitted before calling transform().")
        df_feat = self._engineer_features(df)
        
        for col in CATEGORICAL_COLS:
            if col in df_feat.columns:
                df_feat[col] = df_feat[col].astype(str).replace({'nan': 'Unknown', 'UNK': 'Unknown', 'None': 'Unknown'})
            else:
                df_feat[col] = 'Unknown'

        engineered_num_cols = NUMERICAL_COLS + [
            'diameter_ratio', 'lesion_area', 'max_diameter',
            'symptom_severity_score', 'is_sun_exposed',
            'age_under_30', 'age_30_50', 'age_50_70', 'age_over_70', 'missing_fraction',
            'bcc_score', 'ack_score', 'nev_score', 'sek_score', 'scc_score', 'mel_score', 'abcde_score',
            'log_lesion_area', 'bleed_grew_interaction', 'hurt_bleed_interaction',
            'sun_age_interaction', 'risk_ratio', 'melanoma_high_risk_flag', 'carcinoma_high_risk_flag'
        ]
        num_data = self.num_imputer.transform(df_feat[engineered_num_cols])
        num_scaled = self.scaler.transform(num_data)

        cat_data = self.cat_imputer.transform(df_feat[CATEGORICAL_COLS])
        cat_encoded = self.ohe.transform(cat_data)

        return np.hstack([num_scaled, cat_encoded]).astype(np.float32)


def get_image_transforms(img_size=224, is_train=False):
    """
    Returns image preprocessing and data augmentation pipelines.
    """
    if is_train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
