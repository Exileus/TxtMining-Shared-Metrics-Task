from typing import Dict, List, Union
import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch

srt = ["source", "reference", "translation"]
language_pairs = [
    "cs-en",
    "de-en",
    "en-fi",
    "en-zh",
    "ru-en",
    "zh-en",
]


class TextMiningDataModule(LightningDataModule):
    def __init__(self, batch_size, pair, dims, train_split: float = 0.9):
        super().__init__(dims=dims)
        self.batch_size = batch_size
        self.pair = pair
        self.train_split = train_split

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        scores = {
            pair: pd.read_csv(f"corpus/{self.pair}/scores.csv") for pair in language_pairs
        }
        score = pd.read_csv(f"corpus/{self.pair}/scores.csv")

        if stage == "fit" or stage is None:
            embedding_ref = torch.from_numpy(
                np.load(f"corpus/{self.pair}/laser.reference_embeds.npy")
            ).float()
            embedding_src = torch.from_numpy(
                np.load(f"corpus/{self.pair}/laser.source_embeds.npy")
            ).float()
            embedding_hyp = torch.from_numpy(
                np.load(f"corpus/{self.pair}/laser.translation_embeds.npy")
            ).float()
            train_size = embedding_ref.shape[0]
            train_emb = torch.cat([embedding_src, embedding_ref, embedding_hyp], dim=1)

            train_target = (
                torch.tensor(score["z-score"]).unsqueeze(1).float()
            )

            ds_full = TensorDataset(train_emb, train_target)
            self.ds_train, self.ds_val = random_split(
                ds_full,
                [
                    int(train_size * (self.train_split)),
                    int(train_size - int(train_size * (self.train_split))),
                ],
            )
        if stage == "test" or stage is None:
            embedding_ref = torch.from_numpy(
                np.load(f"testset/{self.pair}/laser.reference_embeds.npy")
            ).float()
            embedding_src = torch.from_numpy(
                np.load(f"testset/{self.pair}/laser.source_embeds.npy")
            ).float()
            embedding_hyp = torch.from_numpy(
                np.load(f"testset/{self.pair}/laser.translation_embeds.npy")
            ).float()
            test_size = embedding_ref.shape[0]
            test_emb = torch.cat([embedding_src, embedding_ref, embedding_hyp], dim=1)
            self.ds_test = TensorDataset(test_emb, torch.rand(test_size, 1))

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(
            self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=12
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=12)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=12)
