
import torch

from . import datasets as D
from .transforms import build_transforms
from .catalog import DatasetCatalog

def build_dataset(dataset_names, data_ids, transform):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_train, coco_2014_val, etc
        transforms (callable): transforms to apply to each (image, target) sample
        is_train (bool): whether to setup the dataset for training or testing
    """
    assert len(dataset_names) == 1, "Only supports one dataset"
    datasets = []

    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    for dataset_name, dic in zip(dataset_names, dataset_dicts):
        assert len(dic), "Dataset '{}' is empty!".format(dataset_name)

        factory = getattr(D, dic["factory"])
        args = dic["args"]
        args['data_ids'] = data_ids
        args['transform'] = transform
        dataset = factory(**args)
        datasets.append(dataset)

    return datasets[0]


def make_data_loader(cfg, is_train):
    if is_train:
        dataset_names = cfg.DATASETS.TRAIN
        shuffle = True
        num_workers = cfg.DATALOADER.NUM_WORKERS
        is_train_transform = True
    else:
        dataset_names = cfg.DATASETS.TEST
        shuffle = True
        num_workers = 0
        is_train_transform = True
    data_ids = cfg.INPUT.FRAME_IDS + cfg.INPUT.CAM_IDS + cfg.INPUT.AUX_IDS
    images_per_batch = cfg.SOLVER.IMS_PER_BATCH

    transform = build_transforms(cfg, is_train=is_train_transform)
    dataset = build_dataset(dataset_names, data_ids, transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=images_per_batch,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=True
    )
    return data_loader