import wandb
import json
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
from architecture import *
from math import sqrt


# ---- PROBES ----

class nn_probe(torch.nn.Module):
    def __init__(self, in_features, hidden_features=64, n_cells=9, n_classes=9):
        super(nn_probe, self).__init__()
        self.linear1 = torch.nn.Linear(in_features, hidden_features)
        self.linear2 = torch.nn.Linear(hidden_features, n_cells * n_classes)

    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        return self.linear2(x)
    

def confusion(pred_cell, y_cell, n_classes=9):
    C = np.zeros((n_classes, n_classes), dtype=int)
    N = len(pred_cell)
    for sample in range(N):
        for p, t in zip(pred_cell[sample], y_cell[sample]):
            C[t, p] += 1   # rows=truth, cols=pred
    return C


def extract_encodings(model, loader, n_cells=9, also_roll_out=True, device=None):
    model.eval()
    encodings, states = [], []
    roll_outs, next_states = [], []
    for xb, ab, yb, sb, sa in loader:
        xb, sb = xb.to(device), sb.to(device)
        if also_roll_out:
            ab, sa = ab.to(device), sa.to(device)
        with torch.no_grad():
            _, _, encoding = model.encoder(xb)
            encoding = encoding.float()
            encodings.append(encoding)
            states.append(sb.reshape(-1, n_cells))
            if also_roll_out:
                roll_out = model.predictor(encoding, ab) > 0.
                roll_outs.append(roll_out.float())
                next_states.append(sa.reshape(-1, n_cells))
    encodings = torch.concatenate(encodings)
    states = torch.concatenate(states)
    if also_roll_out:
        roll_outs = torch.concatenate(roll_outs)
        next_states = torch.concatenate(next_states)
        return TensorDataset(encodings, states), TensorDataset(roll_outs, next_states)
    return TensorDataset(encodings, states)


def retrieve_agent_position(state, n_cells=9):
    """Retrieve agent positions from state tensors."""
    state = state.reshape(-1, int(sqrt(n_cells)), int(sqrt(n_cells)))
    if n_cells == 9 or n_cells == 16: # 8-game or 15-game
        return torch.nonzero(state == 0, as_tuple=False)
    return torch.nonzero((state == 3) | (state == 4), as_tuple=False)  # ice-slider
    

def train_probe(probe, enc_loader, n_cells=9, n_classes=9, epochs=15, lr=1e-3, weight_decay=0., 
                agent_position=False, probe_type="linear", grid_size=None, in_channels=None):
    if agent_position and probe_type == "conv":
        raise ValueError("Agent position probe not supported for convolutional probe_type")
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    grid_size = grid_size or int(sqrt(n_cells))
    for _ in range(epochs):
        for encodings, states in enc_loader:
            if probe_type == "conv":
                if in_channels is None:
                    raise ValueError("in_channels must be provided for convolutional probe training")
                encodings = encodings.view(-1, in_channels, grid_size, grid_size)
                logits = probe(encodings)
                targets = states.view(-1, grid_size, grid_size).long()
                loss = torch.nn.functional.cross_entropy(logits, targets)
            else:
                logits = probe(encodings)
                if agent_position:
                    n_rows = int(sqrt(n_cells))
                    agent_positions = retrieve_agent_position(states, n_cells)
                    loss_row = torch.nn.functional.cross_entropy(logits[:, :n_rows].view(-1, n_rows), agent_positions[:, 1].view(-1))
                    loss_col = torch.nn.functional.cross_entropy(logits[:, n_rows:].view(-1, n_rows), agent_positions[:, 2].view(-1))
                    loss = loss_row + loss_col
                else:
                    loss = torch.nn.functional.cross_entropy(logits.view(-1, n_classes), states.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    return probe


@torch.no_grad()
def evaluate_probe_agent(probe, enc_dataset, n_cells=9):
    n_rows = int(sqrt(n_cells))
    b, y = enc_dataset.tensors
    logits = probe(b)
    pred_row = logits[:, :n_rows].argmax(-1)
    pred_col = logits[:, n_rows:].argmax(-1)
    pred = torch.stack([pred_row, pred_col], dim=1)
    agent_positions = retrieve_agent_position(y, n_cells)[:, 1:]
    correct = (pred == agent_positions).all(dim=1).float()
    acc = correct.mean().item()
    return acc


@torch.no_grad()
def evaluate_probe(probe, enc_dataset, n_cells=9, n_classes=9, Hungarian_matching=True, confusion_matrix=False):
    b, y = enc_dataset.tensors

    # Forward pass
    logits = probe(b).reshape(-1, n_cells, n_classes)
    pred   = logits.argmax(-1)               # (B, n_cells)

    # Accuracy per cell and per grid
    correct_per_cell = (pred == y).float().mean(0)
    grid_acc = (pred == y).all(dim=1).float().mean()

    # mean macro F1 score per cell
    # Compute macro F1 per cell, considering only labels that actually appear in the cell
    f1_per_cell = []
    for i in range(n_cells):
        y_true_i = y[:, i].cpu().numpy()
        y_pred_i = pred[:, i].cpu().numpy()
        present_labels = np.unique(np.concatenate([y_true_i, y_pred_i]))
        f1_per_cell.append(
            f1_score(
                y_true=y_true_i,
                y_pred=y_pred_i,
                labels=present_labels,
                average="macro",
                zero_division=0,
            )
        )

    results = {
        "acc_per_cell": correct_per_cell.mean().item(),
        "grid_acc": grid_acc.item(),
        "f1_per_cell": np.mean(f1_per_cell).item()
    }

    if confusion_matrix:
        results["confusion_matrix"] = confusion(pred.cpu().numpy(), y.cpu().numpy(), n_classes=n_classes)

    if not Hungarian_matching:
        return results

    # Optional Hungarian matching
    pred, y = pred.cpu(), y.cpu()
    logits = logits.cpu().numpy()
    pred2 = np.zeros_like(pred.numpy())

    for i in range(b.size(0)):
        # cost matrix for Hungarian algorithm: shape (n_cells, n_cells)
        cost = -logits[i]  # maximize logits = minimize negative
        rows, cols = linear_sum_assignment(cost)
        pred2[i, rows] = cols

    pred2 = torch.tensor(pred2)
    Hungarian_per_cell = (pred2 == y).float().mean(0)
    Hungarian_acc = (pred2 == y).all(dim=1).float().mean()

    results.update({
        "Hungarian_acc_per_cell": Hungarian_per_cell.mean().item(),
        "Hungarian_grid_acc": Hungarian_acc.item(),
    })

    if confusion_matrix:
        results["Hungarian_confusion_matrix"] = confusion(pred2, y, n_classes=n_classes)

    return results


@torch.no_grad()
def evaluate_conv_probe(probe, enc_dataset, n_cells=9, n_classes=9, confusion_matrix=False, in_channels=None):
    if in_channels is None:
        raise ValueError("in_channels must be provided for convolutional probe evaluation")
    b, y = enc_dataset.tensors
    grid_size = int(sqrt(n_cells))

    encodings = b.view(-1, in_channels, grid_size, grid_size)
    logits = probe(encodings)
    preds = logits.argmax(dim=1).view(-1, n_cells)
    y = y.view(-1, n_cells)

    correct_per_cell = (preds == y).float().mean(0)
    grid_acc = (preds == y).all(dim=1).float().mean()

    f1_per_cell = []
    for i in range(n_cells):
        y_true_i = y[:, i].cpu().numpy()
        y_pred_i = preds[:, i].cpu().numpy()
        present_labels = np.unique(np.concatenate([y_true_i, y_pred_i]))
        f1_per_cell.append(
            f1_score(
                y_true=y_true_i,
                y_pred=y_pred_i,
                labels=present_labels,
                average="macro",
                zero_division=0,
            )
        )

    results = {
        "acc_per_cell": correct_per_cell.mean().item(),
        "grid_acc": grid_acc.item(),
        "f1_per_cell": np.mean(f1_per_cell).item()
    }

    if confusion_matrix:
        results["confusion_matrix"] = confusion(preds.cpu().numpy(), y.cpu().numpy(), n_classes=n_classes)

    return results



def apply_probe(model, train_loader, val_loader, use_conv_probe=False,
                Hungarian_matching=False, confusion_matrix=False, n_cells=9, n_classes=9,
                device=None, epochs=15, lr=1e-2, weight_decay=1e-3):
    encodings_train_ds = extract_encodings(model, train_loader, n_cells=n_cells, also_roll_out=False, device=device)
    encodings_val_ds, roll_outs_val_ds = extract_encodings(model, val_loader, n_cells=n_cells, also_roll_out=True, device=device)
    enc_train_loader = DataLoader(encodings_train_ds, batch_size=train_loader.batch_size)
    n_bits = model.encoder.n_bits
    grid_size = int(sqrt(n_cells))
    if use_conv_probe:
        if n_bits % n_cells != 0:
            raise ValueError(f"Cannot build conv probe: n_bits {n_bits} is not divisible by n_cells {n_cells}")
        in_channels = n_bits // n_cells
        probe = torch.nn.Conv2d(in_channels, n_classes, kernel_size=1).to(device)
        probe = train_probe(probe, enc_train_loader, n_cells=n_cells, n_classes=n_classes, epochs=epochs, lr=lr, weight_decay=weight_decay,
                            probe_type="conv", grid_size=grid_size, in_channels=in_channels)
        linear_results = evaluate_conv_probe(probe, encodings_val_ds, n_cells=n_cells, n_classes=n_classes, confusion_matrix=confusion_matrix, in_channels=in_channels)
        roll_out_results = evaluate_conv_probe(probe, roll_outs_val_ds, n_cells=n_cells, n_classes=n_classes, confusion_matrix=confusion_matrix, in_channels=in_channels)
    else:
        probe = torch.nn.Linear(n_bits, n_cells*n_classes).to(device)
        probe = train_probe(probe, enc_train_loader, n_cells=n_cells, n_classes=n_classes, epochs=epochs, lr=lr, weight_decay=weight_decay)
        linear_results = evaluate_probe(probe, encodings_val_ds, n_cells=n_cells, n_classes=n_classes, Hungarian_matching=Hungarian_matching, confusion_matrix=confusion_matrix)
        roll_out_results = evaluate_probe(probe, roll_outs_val_ds, n_cells=n_cells, n_classes=n_classes, Hungarian_matching=Hungarian_matching, confusion_matrix=confusion_matrix)
    agent_position_probe = torch.nn.Linear(n_bits, 2*grid_size).to(device)
    agent_position_probe = train_probe(agent_position_probe, enc_train_loader, n_cells=n_cells, n_classes=n_classes, epochs=epochs, lr=lr, weight_decay=weight_decay, agent_position=True)
    linear_result_agent = evaluate_probe_agent(agent_position_probe, encodings_val_ds, n_cells=n_cells)
    roll_out_result_agent = evaluate_probe_agent(agent_position_probe, roll_outs_val_ds, n_cells=n_cells)
    return linear_results, roll_out_results, linear_result_agent, roll_out_result_agent
    #if also_neural:
    #    probe = nn_probe(n_bits, n_cells=n_cells).to(device)
    #    probe = train_probe(probe, enc_train_loader, epochs=epochs, lr=lr, weight_decay=weight_decay)
    #    nn_results = evaluate_probe(probe, encodings_val_ds, Hungarian_matching=Hungarian_matching, confusion_matrix=confusion_matrix)
    #    nn_roll_out_results = evaluate_probe(probe, roll_outs_val_ds, Hungarian_matching=Hungarian_matching, confusion_matrix=confusion_matrix)
    #    return linear_results, roll_out_results, nn_results, nn_roll_out_results



# ---- W&B ----

def set_wandb_runs(config, trial, args, best, repeat, id_list):
    """Set up a wandb run for the given trial.
    Parameters:
        config (dict): The configuration dictionary.
        trial (optuna.Trial): The Optuna trial object.
        args (argparse.Namespace): The command-line arguments.
        best (bool): True if this is the best run, False otherwise.
        repeat (int): The repeat number.
        id_list (list): A list to store the run IDs.
    """
    config["trial number"] = trial.number
    config["repeat"] = repeat
    run = wandb.init(
        project=args.experimentname,
        config=config,
        group=args.group + "_" + args.device,
        name=f'trial_{trial.number}_{repeat}' if not best else f'best_{repeat}',
        job_type='Trials' if not best else 'Best',
        reinit=True
    )
    id_list.append(run.id)
    return run

def wandb_prune_run(run, id_list=[], experiment_name=None):
    wandb.summary["pruned"] = True
    run.finish()
    for run_id in id_list:
        run = wandb.init(id=run_id, project=experiment_name, resume="must")
        wandb.summary["pruned"] = True

# ---- JSON ----

def write_results_to_file(results_dict, results_file):
    """Write the results dictionary to a JSON file."""
    try:
        with open(results_file, "r") as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    data.append(results_dict)
    with open(results_file, "w") as f:
        json.dump(data, f, indent=4) # Save back to the file


def get_hyperparameters(results_file, experiment_name, group):
    """Retrieve hyperparameters from a JSON file based on the experiment name and group."""
    try:
        with open(results_file, "r") as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    for results_dict in data:
        if results_dict['experiment'] == experiment_name and results_dict['group'] == group:
            hyperparams = results_dict['hyperparameters']
            break
    if 'hyperparams' not in locals():
        raise ValueError(f"No hyperparameters found for experiment {experiment_name} and group {group}.")
    return hyperparams


# ---- Device ----
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
