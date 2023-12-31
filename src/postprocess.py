
"""Post-processing """
import os

import numpy as np
import matplotlib.pyplot as plt


def plot_train_loss(train_loss, plot_dir, epochs, net_name):
    """Plot change of loss during training"""
    plt.plot(list(range(epochs)), train_loss)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("train_loss")
    plt.savefig(f"{plot_dir}/{net_name}_train_loss.png")
    np.savetxt(f"{plot_dir}/{net_name}_train_loss.txt", train_loss)
    plt.close()


def error(y_true, y_predict):
    relative_error = np.average(np.abs((y_predict - y_true)) / y_true)
    return relative_error


def plot_cae_prediction(cae_encoded, cae_predict, true_data, plot_dir, time_size):
    """Plot cae prediction"""
    # prepare file
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    # cae_prediction
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.title("true time=600")
    plt.contourf(true_data[600])
    plt.subplot(2, 3, 2)
    plt.title("true time=1200")
    plt.contourf(true_data[1200])
    plt.subplot(2, 3, 3)
    plt.title("true time=1786")
    plt.contourf(true_data[-1])
    plt.subplot(2, 3, 4)
    plt.title("cae_predict time=600")
    plt.contourf(cae_predict[600])
    plt.subplot(2, 3, 5)
    plt.title("cae_predict time=1200")
    plt.contourf(cae_predict[1200])
    plt.subplot(2, 3, 6)
    plt.title("cae_predict time=1786")
    plt.contourf(cae_predict[-1])
    plt.savefig(f"{plot_dir}/cae_prediction.png")
    plt.close()

    # relative_error
    time_true = np.arange(0, time_size)
    cae_error = np.zeros(time_size)
    for time in np.arange(time_size):
        cae_error[time] = error(true_data[time], cae_predict[time])

    plt.plot(time_true, cae_error, "k-")
    plt.title("relative error")
    plt.ylabel("error")
    plt.xlabel("t")
    plt.savefig(f"{plot_dir}/cae_error.png")
    plt.close()

    # save prediction
    np.save(f"{plot_dir}/cae_encoded.npy", np.squeeze(cae_encoded))

    np.save(f"{plot_dir}/cae_predict.npy", cae_predict)
    np.save(f"{plot_dir}/cae_error.npy", cae_error)


def plot_cae_transformer_prediction(
    lstm_latent, cae_lstm_predict, true_data, plot_dir, time_size, time_window
):
    """Plot prediction"""
    # prepare file
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    # cae_lstm_prediction
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.title("true time=600")
    plt.contourf(true_data[600])
    plt.subplot(2, 3, 2)
    plt.title("true time=1200")
    plt.contourf(true_data[1200])
    plt.subplot(2, 3, 3)
    plt.title("true time=1786")
    plt.contourf(true_data[-1])
    plt.subplot(2, 3, 4)
    plt.title("cae_transformer_predict time=600")
    plt.contourf(cae_lstm_predict[600 - time_window])
    plt.subplot(2, 3, 5)
    plt.title("cae_transformer_predict time=1200")
    plt.contourf(cae_lstm_predict[1200 - time_window])
    plt.subplot(2, 3, 6)
    plt.title("cae_transformer_predict time=1786")
    plt.contourf(cae_lstm_predict[-1])
    plt.savefig(f"{plot_dir}/cae_transformer_prediction.png")
    plt.close()

    # relative_error
    time_true = np.arange(0, time_size)
    time_predict = time_true[time_window:]

    cae_lstm_error = np.zeros(time_size - time_window)

    for time in np.arange(time_size - time_window):
        cae_lstm_error[time] = error(
            true_data[time + time_window], cae_lstm_predict[time]
        )

    plt.plot(time_predict, cae_lstm_error, "k-")
    plt.title("relative error")
    plt.ylabel("error")
    plt.xlabel("t")
    plt.savefig(f"{plot_dir}/cae_transformer_error.png")
    plt.close()

    # save prediction
    np.save(f"{plot_dir}/transformer_latent.npy", lstm_latent.asnumpy())
    np.save(f"{plot_dir}/cae_transformer_predict.npy", cae_lstm_predict)
    np.save(f"{plot_dir}/cae_transformer_error.npy", cae_lstm_error)
