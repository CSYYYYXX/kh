import os
import time
import argparse
import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import yaml
from paddle.io import DataLoader

from sympy import Q
from src.dataset import create_transformer_dataset
from src import plot_train_loss
from src.model_transformer import Informer
from cae_prediction import cae_prediction


np.random.seed(42)
paddle.seed(42)

def load_yaml_config(config_file_path):
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class CustomWithLossFunction(nn.Layer):
    def __init__(self, model, loss_fn, args):
        super(CustomWithLossFunction, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.args = args

    def forward(self, seq_x, seq_y):
        # Convert data to paddle tensors
        batch_x = paddle.to_tensor(seq_x, dtype='float32')
        batch_y = paddle.to_tensor(seq_y, dtype='float32')

        if self.args["padding"] == 0:
            dec_inp = paddle.zeros((batch_y.shape[0], self.args["pred_len"], batch_y.shape[-1]), dtype='float32')
        else:
            dec_inp = paddle.ones((batch_y.shape[0], self.args["pred_len"], batch_y.shape[-1]), dtype='float32')

        dec_inp = paddle.concat([batch_x[:, -self.args["label_len"]:, :], dec_inp], axis=1).astype('float32')

        outputs = self.model(batch_x, dec_inp)
        batch_y = batch_y[:, -self.args["pred_len"]:, :]

        return self.loss_fn(outputs, batch_y)


def train():
    # prepare params
    config = load_yaml_config(args.config_file_path)
    data_params = config["transformer_data"]
    model_params = config["transformer"]
    optimizer_params = config["transformer_optimizer"]

    # prepare summary file
    summary_dir = optimizer_params["summary_dir"]
    ckpt_dir = os.path.join(summary_dir, "ckpt")

    # prepare model
    model = Informer(**model_params)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(parameters=model.parameters(), learning_rate=optimizer_params["lr"], weight_decay=optimizer_params["weight_decay"])

    # prepare dataset
    latent_true = cae_prediction(args.config_file_path)
    dataset, _ = create_transformer_dataset(
        latent_true,
        data_params["batch_size"],
        data_params["time_size"],
        data_params["latent_size"],
        data_params["time_window"],
        data_params["gaussian_filter_sigma"],
    )

    time_now = time.time()
    loss_net = CustomWithLossFunction(model, loss_fn, data_params)

    for epoch in range(optimizer_params["epochs"]):
        print(f">>>>>>>>>>>>>>>>>>>>Train_{epoch}<<<<<<<<<<<<<<<<<<<<")
        ts = time.time()
        model.train()

        total_loss = 0.0
        num_batches = len(dataset)

        for i, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=True)):
            seq_x, seq_y = batch
            seq_x, seq_y = seq_x.numpy(), seq_y.numpy()

            optimizer.clear_grad()

            loss = loss_net(seq_x, seq_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.numpy()

            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{optimizer_params['epochs']}], Batch [{i+1}/{num_batches}], Loss: {loss.numpy()}")

        print(f"Train Time Cost: {time.time() - ts}")
        print(f"Average Loss for Epoch {epoch+1}: {total_loss / num_batches}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="transformer net for KH")
    parser.add_argument(
        "--mode",
        type=str,
        default="GRAPH",
        choices=["GRAPH", "PYNATIVE"],
        help="Context mode, support 'GRAPH', 'PYNATIVE'",
    )
    parser.add_argument(
        "--save_graphs",
        type=bool,
        default=False,
        choices=[True, False],
        help="Whether to save intermediate compilation graphs",
    )
    parser.add_argument("--save_graphs_path", type=str, default="./graphs")
    parser.add_argument(
        "--device_target",
        type=str,
        default="GPU",
        choices=["GPU", "Ascend"],
        help="The target device to run, support 'Ascend', 'GPU'",
    )
    parser.add_argument(
        "--device_id", type=int, default=0, help="ID of the target device"
    )
    parser.add_argument("--config_file_path", type=str, default="./config.yaml")
    args = parser.parse_args()

    # Check and set the device to use
    paddle.set_device(args.device_target)

    print(f"pid: {os.getpid()}")
    train()

