import lightning.pytorch as pl
from torch.utils.data import DataLoader
import hydra
from functools import partial

class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, train_dataset, val_dataset, test_dataset, collate_fn, pin_memory=True):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.collate_fn = hydra.utils.get_method(collate_fn)
        self.collate_fn = partial(self.collate_fn)
        self.pin_memory = pin_memory

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, shuffle=True, persistent_workers=self.num_workers > 0, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, shuffle=False, persistent_workers=self.num_workers > 0, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, shuffle=False, persistent_workers=self.num_workers > 0, pin_memory=self.pin_memory)

    @property
    def panel_edge_type_indices(self):
        return self.garment_tokenizer.panel_edge_type_indices