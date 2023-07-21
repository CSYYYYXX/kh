import os
import time
import argparse
import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.io import DataLoader

from src import create_cae_dataset, CaeNet, plot_train_loss
import yaml

np.random.seed(0)
paddle.seed(0)


def load_yaml_config(config_file_path):
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def cae_train():
    """CAE net train process"""
    # prepare params
    config = load_yaml_config(args.config_file_path)
    data_params = config["cae_data"]
    model_params = config["cae_model"]
    optimizer_params = config["cae_optimizer"]

    # prepare summary file
    summary_dir = optimizer_params["summary_dir"]
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    ckpt_dir = os.path.join(summary_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # prepare model
    cae = CaeNet(model_params["data_dimension"], model_params["conv_kernel_size"],
                 model_params["maxpool_kernel_size"], model_params["maxpool_stride"],
                 model_params["encoder_channels"], model_params["decoder_channels"],
                 model_params["channels_dense"])
    loss_fn = nn.MSELoss()
    cae_opt = optim.Adam(parameters=cae.parameters(), learning_rate=optimizer_params["lr"],
                         weight_decay=optimizer_params["weight_decay"])

    device = paddle.set_device('gpu') if paddle.is_compiled_with_cuda() else paddle.set_device('cpu')
    cae = cae.to(device)

    # Define forward function
    def forward_fn(data, label):
        logits = cae(data)
        loss = loss_fn(logits, label)
        return loss

    print_freq = 10

    # prepare dataset
    cae_dataset, _ = create_cae_dataset(data_params["data_path"], data_params["batch_size"])
    train_loader = cae_dataset

    print(f"====================Start CAE train=======================")
    train_loss = []
    for epoch in range(1, optimizer_params["epochs"] + 1):
        local_time_beg = time.time()
        cae.train()
        epoch_train_loss = 0
        for i, batch_data in enumerate(train_loader):
            data = paddle.to_tensor(batch_data[0])
            label = paddle.to_tensor(batch_data[1])
            data = paddle.to_tensor(data, place=device)
            label = paddle.to_tensor(label, place=device)
            cae_opt.clear_grad()
            loss = forward_fn(data, label)
            loss.backward()
            cae_opt.step()
            epoch_train_loss += loss.item()

            if i % print_freq == 0:
                print(f"Epoch [{epoch}/{optimizer_params['epochs']}], Step [{i}/{len(train_loader)}], Loss: {loss.item()}")

        train_loss.append(epoch_train_loss)
        print(f"epoch: {epoch} train loss: {epoch_train_loss} epoch time: {time.time() - local_time_beg:.2f}s")

        if epoch % optimizer_params["save_ckpt_interval"] == 0:
            paddle.save(cae.state_dict(), f"{ckpt_dir}/cae_{epoch}.ckpt")
    print(f"=====================End CAE train========================")
    plot_train_loss(train_loss, summary_dir, optimizer_params["epochs"], "cae")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cae net for KH')
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument("--save_graphs", type=bool, default=False, choices=[True, False],
                        help="Whether to save intermediate compilation graphs")
    parser.add_argument("--save_graphs_path", type=str, default="./graphs")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
    parser.add_argument("--config_file_path", type=str, default="./config.yaml")
    args = parser.parse_args()


    print(f"pid: {os.getpid()}")
    cae_train()
