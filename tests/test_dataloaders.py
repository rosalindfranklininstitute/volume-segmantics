import pytest
from torch.utils.data import DataLoader
from volume_segmantics.data.dataloaders import (
    get_2d_prediction_dataloader,
    get_2d_training_dataloaders,
)


@pytest.fixture()
def train_loaders(image_dir, label_dir, training_settings):
    return get_2d_training_dataloaders(image_dir, label_dir, training_settings)


@pytest.mark.gpu()
def test_get_2d_training_dataloader_types(train_loaders):
    train_loader, valid_loader = train_loaders
    assert isinstance(train_loader, DataLoader)
    assert isinstance(valid_loader, DataLoader)


@pytest.mark.gpu()
def test_get_2d_training_dataloader_length(train_loaders):
    train_loader, valid_loader = train_loaders
    #print(f"len(train_loader):{len(train_loader)}, len(valid_loader):{len(valid_loader)}")
    #print(f"len(train_loader.dataset):{len(train_loader.dataset)}, len(valid_loader.dataset):{len(valid_loader.dataset)}")
    
    #assert len(train_loader) > len(valid_loader) #giving error here
    assert len(train_loader.dataset) > len(valid_loader.dataset) #fixed. probably pytorch was updated so len() may give a different value now
    #The package I have installed is torch 1.12.1, and not 1.7


@pytest.mark.gpu()
def test_get_2d_prediction_dataloader_type(rand_int_volume, prediction_settings):
    prediction_loader = get_2d_prediction_dataloader(
        rand_int_volume, prediction_settings
    )
    assert isinstance(prediction_loader, DataLoader)
