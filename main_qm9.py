from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, global_add_pool

import tqdm
import numpy as np
import matplotlib.pyplot as plt

from grokfast import *


class ExampleNet(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features):
        super().__init__()
        conv1_net = nn.Sequential(
            nn.Linear(num_edge_features, 32),
            nn.ReLU(),
            nn.Linear(32, num_node_features * 32),
        )
        conv2_net = nn.Sequential(
            nn.Linear(num_edge_features, 32), nn.ReLU(), nn.Linear(32, 32 * 16)
        )
        self.conv1 = NNConv(num_node_features, 32, conv1_net)
        self.conv2 = NNConv(32, 16, conv2_net)
        self.fc_1 = nn.Linear(16, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, data):
        batch, x, edge_index, edge_attr = (
            data.batch,
            data.x,
            data.edge_index,
            data.edge_attr,
        )
        # First graph conv layer
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        # Second graph conv layer
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = global_add_pool(x, batch)
        x = F.relu(self.fc_1(x))
        output = self.out(x)
        return output


def L2(model):
    L2_ = 0.0
    for p in model.parameters():
        L2_ += torch.sum(p**2)
    return L2_


def rescale(model, alpha):
    for p in model.parameters():
        p.data = alpha * p.data


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    alpha = args.init_scale

    # size = 1000
    epochs = int(100 * 50000 / args.size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the QM9 small molecule dataset
    dset = QM9(".")
    dset = dset[: args.size]
    train_set, test_set = random_split(dset, [int(args.size / 2), int(args.size / 2)])
    trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    # initialize a network
    qm9_node_feats, qm9_edge_feats = 11, 4
    net = ExampleNet(qm9_node_feats, qm9_edge_feats)

    # initialize an optimizer with some reasonable parameters
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    target_idx = 1  # index position of the polarizability label
    net.to(device)

    rescale(net, alpha)
    L2_ = L2(net)

    train_best = 1e10
    test_best = 1e10

    train_losses, test_losses, train_avg_losses, test_avg_losses = [], [], [], []
    step = 0
    grads = None

    for total_epochs in tqdm.trange(epochs):
        epoch_loss = 0
        total_graphs_train = 0

        for batch in trainloader:
            net.train()
            batch.to(device)
            optimizer.zero_grad()
            output = net(batch)
            loss = F.mse_loss(output, batch.y[:, target_idx].unsqueeze(1))
            epoch_loss += loss.item() * batch.num_graphs
            total_graphs_train += batch.num_graphs

            loss.backward()

            #######

            trigger = False

            if args.filter == "none":
                pass
            elif args.filter == "ma":
                grads = gradfilter_ma(
                    net,
                    grads=grads,
                    window_size=args.window_size,
                    lamb=args.lamb,
                    trigger=trigger,
                )
            elif args.filter == "ema":
                grads = gradfilter_ema(
                    net, grads=grads, alpha=args.alpha, lamb=args.lamb
                )
            elif args.filter == "kal":
                grads = gradfilter_kalman(
                    net,
                    grads=grads,
                    process_noise=args.process_noise,
                    measurement_noise=args.measurement_noise,
                    lamb=args.lamb,
                )
            else:
                raise ValueError(f"Invalid gradient filter type `{args.filter}`")

            #######

            optimizer.step()

            train_losses.append(loss.item())

            step += 1

        train_avg_loss = epoch_loss / total_graphs_train
        if train_avg_loss < train_best:
            train_best = train_avg_loss
        train_avg_losses.append(train_avg_loss)

        #######

        test_loss = 0
        total_graphs_test = 0

        net.eval()

        for batch in testloader:
            batch.to(device)
            output = net(batch)
            loss = F.mse_loss(output, batch.y[:, target_idx].unsqueeze(1))
            test_loss += loss.item() * batch.num_graphs
            total_graphs_test += batch.num_graphs
            test_losses.append(loss.item())

        test_avg_loss = test_loss / total_graphs_test
        if test_avg_loss < test_best:
            test_best = test_avg_loss
        test_avg_losses.append(test_avg_loss)

        #######

        tqdm.tqdm.write(
            f"Epochs: {total_epochs} | epoch avg. loss: {train_avg_loss:.3f} | "
            f"test avg. loss: {test_avg_loss:.3f}"
        )

        if (total_epochs + 1) % 100 == 0 or total_epochs == epochs - 1:

            plt.plot(np.arange(len(train_avg_losses)), train_avg_losses, label="train")
            plt.plot(np.arange(len(train_avg_losses)), test_avg_losses, label="val")
            plt.legend()
            plt.title("QM9 Molecule Isotropic Polarizability Prediction")
            plt.xlabel("Optimization Steps")
            plt.ylabel("MSE Loss")
            plt.yscale("log", base=10)
            plt.xscale("log", base=10)
            plt.ylim(1e-4, 100)
            plt.grid()
            plt.savefig(f"results/qm9_loss_{args.label}.png", dpi=150)
            plt.close()

            torch.save(
                {
                    "its": np.arange(len(train_losses)),
                    "its_avg": np.arange(len(train_avg_losses)),
                    "train_acc": None,
                    "train_loss": train_losses,
                    "train_avg_loss": train_avg_losses,
                    "val_acc": None,
                    "val_loss": test_losses,
                    "val_avg_loss": test_avg_losses,
                    "train_best": train_best,
                    "val_best": test_best,
                },
                f"results/qm9_{args.label}.pt",
            )

    #######

    fig, ax = plt.subplots(1, 1, figsize=(4.2, 4.2))

    ax.plot(
        (np.arange(len(test_losses)) + 1)[::20],
        np.mean(np.array(test_losses).reshape(-1, 20), axis=1),
        color="#ff7f0e",
    )
    ax.plot(
        (np.arange(len(train_losses)) + 1)[::20],
        np.mean(np.array(train_losses).reshape(-1, 20), axis=1),
        color="#1f77b4",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 1000)

    ax.set_ylabel("MSE", fontsize=15)
    ax.text(1, 0.003, r"$\alpha=3$", fontsize=15)
    ax.set_ylim(1e-3, 1e2)
    ax.grid()

    fig.savefig(f"results/qm9_grok_{args.label}.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--label", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument(
        "--init_scale", type=float, default=3.0
    )  # init_scale 1.0 no grokking / init_scale 3.0 grokking

    # Grokfast
    parser.add_argument(
        "--filter", type=str, choices=["none", "ma", "ema", "kal"], default="none"
    )
    parser.add_argument("--process_noise", type=float, default=1e-4)
    parser.add_argument("--measurement_noise", type=float, default=1e-2)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--lamb", type=float, default=5.0)
    args = parser.parse_args()

    filter_str = ("_" if args.label != "" else "") + args.filter
    window_size_str = f"_w{args.window_size}"
    alpha_str = f"_a{args.alpha:.3f}".replace(".", "")
    lamb_str = f"_l{args.lamb:.2f}".replace(".", "")

    model_suffix = f"size{args.size}_alpha{args.init_scale:.4f}"

    if args.filter == "none":
        filter_suffix = ""
    elif args.filter == "ma":
        filter_suffix = window_size_str + lamb_str
    elif args.filter == "ema":
        filter_suffix = alpha_str + lamb_str
    elif args.filter == "kal":
        filter_suffix = (
            f"_p{args.process_noise:.1e}_m{args.measurement_noise:.1e}".replace(".", "")
            + lamb_str
        )
    else:
        raise ValueError(f"Unrecognized filter type {args.filter}")

    optim_suffix = ""
    if args.weight_decay != 0:
        optim_suffix = optim_suffix + f"_wd{args.weight_decay:.1e}".replace(".", "")
    if args.lr != 1e-3:
        optim_suffix = optim_suffix + f"_lrx{int(args.lr / 1e-3)}"

    args.label = args.label + model_suffix + filter_str + filter_suffix + optim_suffix
    print(f"Experiment results saved under name: {args.label}")

    main(args)
