
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
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        shuffle = True
        data_ids = cfg.INPUT.FRAME_IDS + cfg.INPUT.CAM_IDS
    else:
        dataset_names = cfg.DATASETS.TEST
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        shuffle = False
        data_ids = cfg.INPUT.FRAME_IDS + cfg.INPUT.CAM_IDS

    transform = build_transforms(cfg, is_train=is_train)
    dataset = build_dataset(dataset_names, data_ids, transform)
    print(cfg.DATALOADER.NUM_WORKERS)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=images_per_batch,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=True
    )
    return data_loader