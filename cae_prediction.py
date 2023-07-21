"""prediction process"""
import os
import argparse
import numpy as np

import paddle
from paddle.static import InputSpec, load_inference_model

from src import CaeNet, create_cae_dataset, plot_cae_prediction
import yaml

np.random.seed(0)
paddle.seed(0)

def load_yaml_config(config_file_path):
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def cae_prediction(config_file_path):
    """Process of prediction with cae net"""
    # prepare params
    config = load_yaml_config(config_file_path)
    data_params = config["cae_data"]
    model_params = config["cae_model"]
    prediction_params = config["prediction"]

    # prepare network
    cae = CaeNet(model_params["data_dimension"], model_params["conv_kernel_size"], model_params["maxpool_kernel_size"],
                 model_params["maxpool_stride"], model_params["encoder_channels"], model_params["decoder_channels"],
                 model_params["channels_dense"])

    # load checkpoint
    cae_state_dict = paddle.load(prediction_params["cae_ckpt_path"])
    cae.set_state_dict(cae_state_dict)

    # prepare dataset
    _, true_data = create_cae_dataset(data_params["data_path"], data_params["batch_size"]) # [time_size, 256, 256]
    data_set = np.expand_dims(true_data, 1).astype(np.float32)

    print(f"=================Start cae prediction=====================")
    encoded = np.zeros((data_params["time_size"], model_params["latent_size"]), dtype=np.float32)
    cae_predict = np.zeros(true_data.shape, dtype=np.float32)
    for i in range(prediction_params["encoder_data_split"]):
        time_predict_start, time_predict_end = \
            prediction_params["encoder_time_spilt"][i], prediction_params["encoder_time_spilt"][i+1]
        encoded[time_predict_start: time_predict_end] = \
            cae.encoder(paddle.to_tensor(data_set[time_predict_start: time_predict_end]))
        cae_predict[time_predict_start: time_predict_end] = \
            np.squeeze(cae(paddle.to_tensor(data_set[time_predict_start: time_predict_end])).numpy())
    print(f"===================End cae prediction====================")
    plot_cae_prediction(encoded, cae_predict, true_data,
                        prediction_params["prediction_result_dir"], data_params["time_size"])
    return encoded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cae prediction')
    parser.add_argument("--config_file_path", type=str, default="./config.yaml")
    args = parser.parse_args()

    paddle.set_device("gpu" if paddle.is_compiled_with_cuda() else "cpu")

    print(f"pid:{os.getpid()}")
    cae_prediction(args.config_file_path)

