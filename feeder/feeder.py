import loaderx
import numpy as np


class Dataset(loaderx.Dataset):
    """
    A class that extends the Dataset class from loaderx.
    This class can be used to create datasets for machine learning tasks.
    """

    def __init__(self, root_dir, mode):
        self.root_dir = root_dir
        self.data, self.label = self.load_data(mode)

    def load_data(self, mode):
        """
        Load data from the specified file in the root directory.
        """
        image_file_path = f"{self.root_dir}/{mode}_images.npy"
        label_file_path = f"{self.root_dir}/{mode}_labels.npy"
        return np.load(image_file_path), np.load(label_file_path)

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the specified index.
        """
        if idx < 0 or idx >= len(self.data):
            raise IndexError("Index out of range")
        return self.data[idx], self.label[idx]
