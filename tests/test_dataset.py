import pytest
import torch
import numpy as np
from PIL import Image
from src.core.deep_learning.dataset_emnist import fix_emnist_orientation, to_three_channels, get_emnist_transforms
from src.core.deep_learning.dataset_synthetic import SyntheticPlateDataset, get_synthetic_transforms

def test_fix_emnist_orientation():
    # Create a small non-square PIL image
    img = Image.new("L", (10, 20))
    fixed = fix_emnist_orientation(img)
    # The transpose operation should swap dimensions
    assert fixed.size == (20, 10)

def test_to_three_channels():
    # Dummy single channel tensor
    dummy_gray = torch.randn(1, 32, 32)
    three_chan = to_three_channels(dummy_gray)
    assert three_chan.shape == (3, 32, 32)
    assert torch.equal(three_chan[0], three_chan[1])
    assert torch.equal(three_chan[1], three_chan[2])

def test_synthetic_plate_dataset():
    char_list = ["A", "B", "京"]
    num_samples = 3
    transform = get_synthetic_transforms()
    
    dataset = SyntheticPlateDataset(
        char_list=char_list,
        num_samples_per_class=num_samples,
        size=32,
        transform=transform
    )
    
    # Total samples should be len(char_list) * num_samples
    assert len(dataset) == len(char_list) * num_samples
    
    # Get a sample
    img_tensor, label = dataset[0]
    
    # Verify shape and type
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 32, 32)
    assert 0 <= label < len(char_list)
