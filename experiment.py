import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import optuna
import wandb
import argparse
import time

from architecture import *
from dataset import *
from utils import *
from loss_functions import *

WRITE_FILE = "data/res.json"
READ_FILE = "data/res.json"

def run_epoch_jepa(jepa_model, opt, loader, train=True, prediction_step=False, device=None):
    '''
    Run one epoch training or evaluating the JEPA model.
    '''
    jepa_model.train(train)
    total_loss = 0.0
    total_details = {}
    for xb, ab, yb, _, _ in loader:
        xb, ab, yb = xb.to(device), ab.to(device), yb.to(device)
        if prediction_step and train:
            pred, bits_tgt = jepa_model(xb, ab, yb, compute_losses=False)
            loss = prediction_loss(pred, bits_tgt, jepa_model.cfg.jepa_loss)
            opt.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(jepa_model.parameters(), max_norm=1.0)
            opt.step()
        loss, details, _, _, _ = jepa_model(xb, ab, yb)
        if train:
            opt.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(jepa_model.parameters(), max_norm=1.0)
            opt.step()
        total_loss += loss.item() * xb.size(0)
        total_details = {k: total_details.get(k, 0.0) + details.get(k, 0.0).item() * xb.size(0) for k in details}
    loss = total_loss / len(loader.dataset)
    details = {k: v / len(loader.dataset) for k, v in total_details.items()}
    return loss, details

def run_epoch_encoder(encoder, heads, opt, loader, train=True, device=None, n_classes=9):
    '''
    Run one epoch training or evaluating the encoder with direct supervision heads, for diagnostic purposes.
    '''
    encoder.train(train)
    total_loss = 0.0
    for xb, _, _, sb, _ in loader:
        xb, sb = xb.to(device), sb.to(device)
        logits, probs, bits = encoder(xb)
        preds = heads(probs)
        loss = torch.nn.functional.cross_entropy(preds.view(-1, n_classes), sb.view(-1))
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_loss += loss.item() * xb.size(0)
    total_loss = total_loss / len(loader.dataset)
    return total_loss


def objective(trial, best=False):
    global args, device, train_loader, val_loader
    
    # Define the hyperparameters
    if args.encoderonly:
        temperature = trial.suggest_float("temperature", 0.5, 2.0)
        lr_cnn = trial.suggest_float("lr_cnn", 1e-4, 5e-3, log=True)
        lr_heads = trial.suggest_float("lr_heads", 1e-4, 1e-1, log=True)
        #channels = trial.suggest_int("channels", 8, 24, step=8) 
        #hidden_dim = trial.suggest_int("hidden_dim", 64, 128, step=32)
    else:  
        lambda_pred = 1. 
        lambda_dec = trial.suggest_float("lambda_dec", 0.2, 100.0, log=True) if args.decoder else 0.
        lambda_loc = trial.suggest_float("lambda_loc", 0.2, 10.0, log=True) if not args.nolocality else 0.
        lambda_third = trial.suggest_float("lambda_third", 0.2, 10.0, log=True) if not args.nothird else 0.
        lambda_var = trial.suggest_float("lambda_var", 0.2, 10.0, log=True) if not args.novar else 0.
        lambda_cov = trial.suggest_float("lambda_cov", 0.2, 10.0, log=True) if not args.nocov else 0.
        lambda_kl = trial.suggest_float("lambda_kl", 0.01, 1.0, log=True) if args.variational else 0.
        hinge_std = trial.suggest_float("hinge_std", 0.26, 0.44) if not args.novar else 0.5
        if args.iceslider:
            max_bits_change = trial.suggest_int("max_bits_change", 3, 9) if not args.nolocality else 6
            min_bits_change = 0
        else:
            max_bits_change = trial.suggest_int("max_bits_change", 8, 12) if not args.nolocality else 12
            min_bits_change = trial.suggest_int("min_bits_change", 2, 4) if not args.nolocality else 2
        #temperature = trial.suggest_float("temperature", 0.5, 2.0)
        ema_decay = trial.suggest_float("ema_decay", 0.66, 0.93) if not args.bothbranches else 0.
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        noise_magnitude = trial.suggest_float("noise_magnitude", 0.0, 2.0) if args.variational else 0.
    n_bits = trial.suggest_int("latent_bits", 40, 96, step=8) if args.tunenbits else args.nbits
    
    if args.scheduling:
        mult_temperature = trial.suggest_float("mult_temp", 0.97, 1.03)
        if args.encoderonly:
            mult_lr_cnn = trial.suggest_float("mult_lr_cnn", 0.97, 1.03)
            mult_lr_heads = trial.suggest_float("mult_lr_heads", 0.97, 1.03)
        else:
            mult_noise_magnitude = trial.suggest_float("mult_noise_magnitude", 0.97, 1.03) if args.variational else 0.
            mult_ema_decay = trial.suggest_float("mult_ema", 0.97, 1.03) if not args.bothbranches else 0.
            mult_lambda_dec = trial.suggest_float("mult_lambda_dec", 0.9, 1.0) if (args.decoder or args.variational) else 0.
            mult_lambda_var_cov_third = trial.suggest_float("mult_lambda_var_cov_third", 0.97, 1.03)  if not (args.nothird or args.novar or args.nocov) else 0.
            mult_lambda_loc = trial.suggest_float("mult_lambda_loc", 0.96, 1.0) if not args.nolocality else 0.
            mult_lambda_pred = trial.suggest_float("mult_lambda_pred", 1, 1.1)
            mult_lr = trial.suggest_float("mult_lr", 0.97, 1.03)
            
    if "lambda_var"   == args.ablated_key: lambda_var   = 0.0
    if "lambda_cov"   == args.ablated_key: lambda_cov   = 0.0
    if "lambda_third" == args.ablated_key: lambda_third = 0.0
    if "lambda_loc"   == args.ablated_key: lambda_loc   = 0.0
    if "lambda_dec"   == args.ablated_key: lambda_dec   = 0.0
    if "lambda_kl"    == args.ablated_key: lambda_kl    = 0.0
    if "ema_decay"    == args.ablated_key: ema_decay    = 0.0
    if "lambda_pred"  == args.ablated_key: lambda_pred  = 0.0

    if args.iceslider:
        N_CELLS, N_CLASSES = 64, 4
        OUT_DIM_CNN = 8
        IMG_CHANNELS = 3
        HIDDEN_DIM_FC = False
        CHANNELS = (32, 3)
        HIDDEN_DIM_PRED = 192
    else:
        N_CELLS, N_CLASSES = args.ngame + 1, args.ngame + 1
        OUT_DIM_CNN = 11 if args.ngame == 8 else 14
        IMG_CHANNELS = 1 
        HIDDEN_DIM_FC = 96
        CHANNELS = (8, 16, 32, 32, 16)
        HIDDEN_DIM_PRED = 128

    standard_config = {
        'img_channels': IMG_CHANNELS,
        'out_dim_cnn': OUT_DIM_CNN,
        'hidden_dim_fc': HIDDEN_DIM_FC,
        'intermediate_channels': CHANNELS,
        'hidden_dim_pred': HIDDEN_DIM_PRED,
        'latent_bits': n_bits,
        'lambda_pred': lambda_pred,
        'temperature': 1.0,
        'normalize_cov': True, 
        'dataset_name': args.datasetname,
        'ice_slider': args.iceslider,
        'jepa_loss': args.loss,
        'decoder': args.decoder,
        'use_straight_through': args.straightthrough,
        'both_branches': args.bothbranches,
        'update_target_branch': args.deepcubeai
    }

    id_list = []
    summary = {}
    summary['details_tr'], summary['details_va'] = [], []
    summary['encoding_probe'], summary['rollout_probe'] = [], []
    summary['agent_encoding_probe'], summary['agent_rollout_probe'] = [], []
    summary['runtime'] = []
    pruned = False

    for rep in range(args.repetitions):
        if args.encoderonly:
            config = dict(trial.params)
            model = BitEncoder(IMG_CHANNELS, CHANNELS, OUT_DIM_CNN,  HIDDEN_DIM_FC, n_bits, temperature, noise_magnitude).to(device)
            head =torch.nn.Linear(n_bits, N_CELLS*N_CLASSES).to(device) 
            opt = torch.optim.Adam([
                {'params': model.parameters(), 'lr': lr_cnn},
                {'params': head.parameters(), 'lr': lr_heads},
            ])
        else:
            cfg = JEPABitsConfig(
                **standard_config,
                lambda_var = lambda_var,
                lambda_cov = lambda_cov,
                lambda_loc = lambda_loc,
                lambda_third = lambda_third,
                lambda_dec = lambda_dec,
                lambda_kl=lambda_kl,
                max_bits_change = max_bits_change,
                min_bits_change=min_bits_change,
                hinge_std=hinge_std,
                ema_decay = ema_decay,
                noise_magnitude = noise_magnitude,
            )
            hyperparams = dict(trial.params)
            config = {**standard_config, **hyperparams}
            jepa_model = JEPABits(cfg).to(device)
            opt = torch.optim.Adam(jepa_model.parameters(), lr=lr)
            if args.scheduling:
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, mult_lr)

        print(f"Trial {trial.number}, repetition {rep+1}/{args.repetitions}")
        run = set_wandb_runs(config, trial, args, best, rep, id_list)

        previous_results = []
        train_time, val_time = 0, 0
        for epoch in range(args.epochs):
            if args.encoderonly:
                start_time = time.time()
                tr_loss = run_epoch_encoder(model, head, opt, train_loader, train=True, device=device, n_classes=N_CLASSES)
                end_training_time = time.time()
                train_time += (end_training_time - start_time)
                va_loss = run_epoch_encoder(model, head, opt, val_loader, train=False, device=device, n_classes=N_CLASSES)
                val_time += (time.time() - end_training_time)
                details_tr = {"l_total": tr_loss}
                details_va = {"l_total": va_loss}
            else:
                try:
                    start_time = time.time()
                    tr_loss, details_tr = run_epoch_jepa(jepa_model, opt, train_loader, train=True, prediction_step=args.extrastep, device=device)
                    end_training_time = time.time()
                    train_time += (end_training_time - start_time)
                    va_loss, details_va = run_epoch_jepa(jepa_model, opt, val_loader, train=False, device=device)
                    val_time += (time.time() - end_training_time)
                except Exception as e:
                    print(f"Error in epoch {epoch}: {e}")
                    for name, p in jepa_model.named_parameters():
                        print("weight", name, p.abs().max().item())
                    print()
                    break
            print(f"-------------------------------- EPOCH {epoch} --------------------------------")
            #print(f"Hyperparameters: {jepa_model.cfg}, lr: {opt.param_groups[0]['lr']}")
            print(f"epoch {epoch:02d}: train loss {tr_loss:.4f} | val loss {va_loss:.4f}")
            print(f"  details train: {details_tr}")
            print(f"  details val: {details_va}")
            print()
            
            wandb.log({
                **{f"train/{k}": v for k, v in details_tr.items()},
                **{f"val/{k}": v for k, v in details_va.items()},
            }, step=epoch)
            
            if args.scheduling:
                if args.encoderonly:
                    model.temperature *= mult_temperature
                    opt.param_groups[0]['lr'] *= mult_lr_cnn
                    opt.param_groups[1]['lr'] *= mult_lr_heads
                else:
                    jepa_model.encoder.temperature *= mult_temperature
                    jepa_model.encoder.noise_magnitude *= mult_noise_magnitude
                    jepa_model.cfg.lambda_dec *= mult_lambda_dec
                    jepa_model.cfg.lambda_var *= mult_lambda_var_cov_third
                    jepa_model.cfg.lambda_cov *= mult_lambda_var_cov_third
                    jepa_model.cfg.lambda_third *= mult_lambda_var_cov_third
                    jepa_model.cfg.lambda_loc *= mult_lambda_loc
                    jepa_model.cfg.lambda_pred *= mult_lambda_pred
                    jepa_model.cfg.lambda_kl *= mult_lambda_dec
                    lr_scheduler.step()
                    if jepa_model.encoder_ema is not None:
                        jepa_model.encoder_ema.decay = min(0.9999, jepa_model.encoder_ema.decay * mult_ema_decay)

            # probe
            if epoch % args.interval == 0:
                linear_results, after_results, acc_agent, roll_out_agent = apply_probe(jepa_model, train_loader, val_loader, device=device, 
                                                                                       n_cells=N_CELLS, n_classes=N_CLASSES, use_conv_probe=args.iceslider)
                wandb.log({f"probes/encoding_{k}": v for k, v in linear_results.items()}, step=epoch)
                wandb.log({f"probes/rollout_{k}": v for k, v in after_results.items()}, step=epoch)
                wandb.log({"probes/acc_agent": acc_agent, "probes/roll_out_agent": roll_out_agent}, step=epoch)
                print(f"  encoding results: {linear_results}")
                print(f"  roll-out results: {after_results}")
                print(f"  acc agent: {acc_agent}")
                print(f"  roll-out agent: {roll_out_agent}")
                print()

                #pruning
                if not best and epoch >= args.epochs//2:
                    if (args.iceslider and (linear_results["f1_per_cell"] < 0.6 or after_results['f1_per_cell'] < 0.3)) or \
                            (not args.iceslider and (after_results["acc_per_cell"] < 0.125 or linear_results["acc_per_cell"] < 0.25)):
                        pruned = True
                        wandb_prune_run(run, id_list, args.experimentname)
                        raise optuna.exceptions.TrialPruned(f"Pruned at epoch {epoch} with value {linear_results["f1_per_cell"]:.4f}")

                #early stopping
                value = after_results["f1_per_cell"]
                if epoch//args.interval > 3 and previous_results[-2] > previous_results[-1] > value:
                    print(f"Early stopping at epoch {epoch} with value {value:.4f} " + \
                            f"(lower than previous {previous_results[-1]:.4f} and the one before {previous_results[-2]:.4f})")
                    break
                else:
                    previous_results.append(value)
                    
        summary['runtime'].append({'train_time': train_time, 'val_time': val_time})
        summary['details_tr'].append(details_tr)
        summary['details_va'].append(details_va)
        print(f'Finished trial {trial.number}, repetition {rep+1}/{args.repetitions} with val loss: {va_loss:.4f}')

        #final probe
        linear_results, after_results, acc_agent, roll_out_agent = apply_probe(jepa_model, train_loader, val_loader, 
                                                                               Hungarian_matching=True, device=device, 
                                                                               n_cells=N_CELLS, n_classes=N_CLASSES, 
                                                                               confusion_matrix=True, use_conv_probe=args.iceslider)
        cm = linear_results.pop("confusion_matrix")
        cm_after = after_results.pop("confusion_matrix")
        for k, v in linear_results.items():
            if k in ["grid_acc", "acc_per_cell", "f1_per_cell"]:
                wandb.log({f"probes/encoding_{k}": v}, step=epoch)
            wandb.summary[f"encoding_probe/{k}"] = v
        print(f"Linear probe results: {linear_results}")
        print(f"Acc agent probe: {acc_agent}")
        print("Confusion matrix encoding:")
        print(cm)
        wandb.log({"probes/acc_agent": acc_agent}, step=epoch)
        wandb.summary["encoding_probe/acc_agent"] = acc_agent
        for k, v in after_results.items():
            if k in ["grid_acc", "acc_per_cell", "f1_per_cell"]:
                wandb.log({f"probes/rollout_{k}": v}, step=epoch)
            wandb.summary[f"rollout_probe/{k}"] = v
        print(f"Roll-out results: {after_results}")
        print(f"Roll-out agent probe: {roll_out_agent}")
        print("Confusion matrix roll-out:")
        print(cm_after)
        print()
        wandb.log({"probes/roll_out_agent": roll_out_agent}, step=epoch)
        wandb.summary["rollout_probe/roll_out_agent"] = roll_out_agent
        wandb.summary['time/train'] = train_time
        wandb.summary['time/val'] = val_time 
        summary['encoding_probe'].append(linear_results)
        summary['rollout_probe'].append(after_results)
        summary['agent_encoding_probe'].append({"agent_encoding_probe": acc_agent})
        summary['agent_rollout_probe'].append({"agent_rollout_probe": roll_out_agent})

        
        trial.report(after_results["f1_per_cell"], rep)
        if trial.should_prune():
            pruned = True
            wandb_prune_run(run, id_list, args.experimentname)
            pruned_value = after_results["f1_per_cell"]
            raise optuna.exceptions.TrialPruned(f"Pruned at run {rep} with value {pruned_value:.4f}")
        
        wandb.summary["finished"] = True
        run.finish()

    mean_summary, std_summary = {}, {}
    for name, value_list in summary.items():
        mean_summary[name] = {k: np.mean([d[k] for d in value_list]) for k in value_list[0]}
        std_summary[name] = {k: np.std([d[k] for d in value_list]) for k in value_list[0]}
    
    print(f"===== Final results over {args.repetitions} repetitions, trial {trial.number} =====")
    print(f"mean details val: {mean_summary['details_va']}, std: {std_summary['details_va']}")
    print(f"mean encoding probe: {mean_summary['encoding_probe']}, std: {std_summary['encoding_probe']}")
    print(f"mean rollout probe: {mean_summary['rollout_probe']}, std: {std_summary['rollout_probe']}")
    print(f"mean agent encoding probe: {mean_summary['agent_encoding_probe']}, std: {std_summary['agent_encoding_probe']}")
    print(f"mean agent rollout probe: {mean_summary['agent_rollout_probe']}, std: {std_summary['agent_rollout_probe']}")
    print(f"mean runtime: {mean_summary['runtime']}, std: {std_summary['runtime']}")
    print()
    
    for run_id in id_list:
        run = wandb.init(id=run_id, project=args.experimentname, resume="must")
        for name, value_dict in mean_summary.items():
            for k, v in value_dict.items():
                wandb.summary[f"mean_{name}/{k}"] = v
        for name, value_dict in std_summary.items():
            for k, v in value_dict.items():
                wandb.summary[f"std_{name}/{k}"] = v
        wandb.summary["pruned"] = pruned
        run.finish()

    if best:
        whole_dict = {"experiment": args.experimentname, "group": args.group, "ngame": args.ngame, "iceslider": args.iceslider,
                      "repetitions": args.repetitions, "device": args.device,
                      "batch_size": args.batchsize, "epochs": args.epochs,
                      "prediction_step": args.extrastep,
                      "hyperparameters": hyperparams, "standard_config": standard_config,
                      "mean_summary": mean_summary, "std_summary": std_summary}
        write_results_to_file(whole_dict, WRITE_FILE)
    
    return mean_summary["rollout_probe"]["f1_per_cell"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experimentname', type=str, default='8-game', help='Name of the experiment')
    parser.add_argument('--iceslider', action='store_true', help='Iceslider dataset')
    parser.add_argument('--ngame', type=int, default=8, help='Number of tiles in the game (8 or 15)')
    parser.add_argument('--repetitions', type=int, default=3, help='Number of runs (per trial)')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--batchsize', type=int, default=256, help='Batch size')
    parser.add_argument('--trials', type=int, default=100, help='Number of trials for hyperparameter optimization')
    parser.add_argument('--group', type=str, default='Trials', help='Group name for wandb')
    parser.add_argument('--interval', type=int, default=5, help='Interval of epochs for applying probes and logging')
    parser.add_argument('--nohypertuning', action='store_true', help='Avoid hyperparameter tuning and use stored values')
    parser.add_argument('--nolocality', action='store_true', help='Remove loss for locality bias (default: do not use it)')
    parser.add_argument('--nothird', action='store_true', help='Remove loss for third order correlation')
    parser.add_argument('--novar', action='store_true', help='Do not use variance loss')
    parser.add_argument('--nocov', action='store_true', help='Do not use covariance loss')
    parser.add_argument('--nbits', type=int, default=64, help='Number of bits')
    parser.add_argument('--tunenbits', action='store_true', help='Tune number of bits')
    parser.add_argument('--scheduling', action='store_true', help='Use scheduling for hyperparameters')
    parser.add_argument('--loss', type=str, default='cross-entropy', help='Type of loss')
    parser.add_argument('--noisetype', type=str, default='none', help='Type of noise: none, structured, or gaussian')
    parser.add_argument('--noisestrength', type=float, default=0.5, help='Strength of noise')
    parser.add_argument('--encoderonly', action='store_true', help='Train and evaluate encoder only, with direct supervision')
    parser.add_argument('--extrastep', action='store_true', help='Perform extra next latent prediction step')
    parser.add_argument('--decoder', action='store_true', help='Use decoder for autoencoding loss')
    parser.add_argument('--noreg', action='store_true', help='Do not use our regularization losses (var, cov, third, loc)')
    parser.add_argument('--straightthrough', action='store_true', help='Use straight through for predictor')
    parser.add_argument('--ablation', action='store_true', help='Perform ablation study')
    parser.add_argument('--variational', action='store_true', help='Use variational autoencoder')
    parser.add_argument('--bothbranches', action='store_true', help='Regularization/decoder loss on both branches')
    parser.add_argument('--deepcubeai', action='store_true', help='Use DeepCubeAI world model')
    parser.add_argument('--uniqueimages', action='store_true', help='Use just one image per digit in the 8/15-game dataset')

    args = parser.parse_args()
    device = get_device()
    args.device = str(device)
    print("Device: ", device)
    print("Args: ", args)
    
    if args.deepcubeai:
        args.noreg = True
        args.bothbranches = True
        args.decoder = True
        args.loss = 'mse'
        args.straightthrough = True
    if args.noreg:
        args.novar = True
        args.nocov = True
        args.nothird = True
        args.nolocality = True
    if args.ablation:
        args.nohypertuning = True
    if args.nohypertuning:
        args.repetitions = 10
    args.ablated_key = None
    args.noisestrength = 0. if args.noisetype == 'none' else args.noisestrength

    if args.iceslider: 
        TRAINING_SIZE = 40000
        VALIDATION_SIZE = 10000
        N_REPEAT=2
        MAX_STEPS=20 
        EXCLUDE_DO_NOTHING=True
        MIN_SOL_LENGTH=1
    elif args.ngame == 8:
        TRAINING_SIZE = 30000
        VALIDATION_SIZE = 6000
    elif args.ngame == 15:
        TRAINING_SIZE = 80000
        VALIDATION_SIZE = 10000
    else:
        raise ValueError("ngame must be 8 or 15")

    if args.iceslider:
        train_ds, val_ds, test_ds, args.datasetname = get_dataset_splits_iceslider(TRAINING_SIZE, VALIDATION_SIZE,
                                                                            n_repeat=N_REPEAT, max_steps=MAX_STEPS,
                                                                            exclude_do_nothing=EXCLUDE_DO_NOTHING,
                                                                            min_sol_len=MIN_SOL_LENGTH,
                                                                            noise_std=args.noisestrength,
                                                                            regenerate=False)
    else:
        train_ds, val_ds, test_ds, args.datasetname = get_dataset_splits(TRAINING_SIZE, VALIDATION_SIZE, 
                                                                            noise_type=args.noisetype, noise_magnitude=args.noisestrength, 
                                                                            n_game=args.ngame, unique_images=args.uniqueimages,
                                                                            regenerate=False)
    print(f"Ready. Dataset: train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)}")
    
    if not args.nohypertuning:
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
        study = optuna.create_study(direction='maximize', pruner=pruner)
        study.optimize(objective, n_trials=args.trials)
        best_params = study.best_params
        print("Best hyperparameters:", best_params)
        print("Best validation loss:", study.best_value)

        args.repetitions = 10
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        val_loader = DataLoader(test_ds, batch_size=256)
        objective(optuna.trial.FixedTrial(best_params), best=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        val_loader = DataLoader(test_ds, batch_size=256)
        hyperparams = get_hyperparameters(READ_FILE, args.experimentname, args.group)
        group = args.group
        ADDITIONAL_STRING = "-ablation" if args.ablation else ""
        args.experimentname = f'{args.experimentname}{ADDITIONAL_STRING}'
        
        # Repeat original
        args.group = f'{group}-new'
        print("Using stored hyperparameters:", hyperparams)
        objective(optuna.trial.FixedTrial(hyperparams), best=True)

        if args.ablation:
            if args.scheduling:
                args.scheduling = False
                args.group = f'{group}_no_scheduling'
                print("Ablated scheduling:", hyperparams)
                objective(optuna.trial.FixedTrial(hyperparams), best=True)
                args.scheduling = True
            ablate_keys = ['lambda_var', 'lambda_cov', 'lambda_third', 'lambda_loc', 'lambda_pred', 'ema_decay']
            if args.decoder: ablate_keys.append('lambda_dec')
            if args.variational: ablate_keys.append('lambda_kl')
            for key in ablate_keys:
                args.ablated_key = key
                args.group = f'{group}_no_{key}'
                print("Ablated hyperparameter:", key, hyperparams)
                objective(optuna.trial.FixedTrial(hyperparams), best=True)
