import paddle
from paddle.io import Dataset, DataLoader
from scipy.ndimage import gaussian_filter1d
import numpy as np

class CreateDataset(Dataset):
    """convert raw data into train dataset"""

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


def create_cae_dataset(data_path, batch_size):
    """create cae dataset"""
    true_data = np.load(data_path)
    train_data = np.expand_dims(true_data, 1).astype(np.float32)

    dataset = CreateDataset(train_data, train_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, true_data


def create_transformer_dataset(
    latent_true, batch_size, time_size, latent_size, time_window, gaussian_filter_sigma
):
    """create transformer dataset"""
    latent_true = np.squeeze(latent_true)
    latent_true = latent_true.astype(np.float32)
    encoded_f = np.copy(latent_true).astype(np.float32)

    for i in range(latent_size):
        encoded_f[:, i] = gaussian_filter1d(
            encoded_f[:, i], sigma=gaussian_filter_sigma
        )

    input_seq = np.zeros(
        shape=(time_size - time_window, time_window, latent_size)
    ).astype(np.float32)
    output_seq = np.zeros(shape=(time_size - time_window, 1, latent_size)).astype(
        np.float32
    )

    sample = 0
    for t in range(time_window, time_size):
        input_seq[sample, :, :] = encoded_f[t - time_window : t, :]
        output_seq[sample, 0, :] = encoded_f[t, :]
        sample = sample + 1

    dataset = CreateDataset(input_seq, output_seq)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, input_seq
