import os
import random
import torch
import warnings
import wandb
import numpy as np
from torchdrug import datasets, transforms, models, tasks, core


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def initialize_weights(m):
    # Conv1d init
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    # Linear init
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    # BatchNorm1d init
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)	 


# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
warnings.filterwarnings("ignore")

sweep_configuration = {
    'method': 'grid',
    'parameters':
    {
        'random_seed': {'values': [42]},
        'batch_size': {'values': [64]},  # 8, 16, 32
        'learning_rate': {'values': [1e-3]},  # 1e-3, 5e-4, 1e-4
        'weight_decay': {'values': [1e-4]},  # 1e-3, 5e-4, 1e-4
        # 'target_readout': {'values': ["mean"]},  # "mean", "sum"
        # 'target_short': {'values': [True]},  # False, True
        # 'target_concat': {'values': [False]},  # False, True
        'target_hidden': {'values': [16]},
        'target_layer': {'values': [3]},  # 2, 3, 4, 5
        # 'target_topk': {'values': [32, 64, 128, 256]},
        'target_raidus': {'values': [6.0, 9.0, 12.0, 15.0]},  # 3.0, 6.0, 9.0, 12.0
    }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project='PDBbind Pocket Prediction Sweep')
# sweep_id = "marinehdk/DPDBbind Pocket Prediction Swee/wnnhwtai"


def main():
    wandb.init()
    seed_torch(wandb.config.random_seed)
    # ============================== Dataset Loading ============================== #
    transform = transforms.ProteinView(view="residue", keys="graph1")
    dataset = datasets.PDBbindProtein(protein_method="gearnet", transform=transform, lazy=True)

    train_set, valid_set, test_set = dataset.ccv_split()
    print(f"Train samples: {len(train_set)}, Valid samples: {len(valid_set)}, Test samples: {len(test_set)}")
    # ============================== Model Defining  ============================== #
    # protein_model = models.GearNet(
    #     input_dim=21, hidden_dims=[wandb.config.target_hidden] * wandb.config.target_layer,
    #     num_relation=7, batch_norm=True, short_cut=wandb.config.target_short,
    #     concat_hidden=wandb.config.target_concat, readout=wandb.config.target_readout
    # )
    #
    # protein_model = models.Conv1D(
    #     input_dim=21, hidden_dims=[wandb.config.target_hidden] * wandb.config.target_layer, kernel_size=7, padding=3
    # )
    # protein_model = models.GCN(
    #     input_dim=21, hidden_dims=[wandb.config.target_hidden] * wandb.config.target_layer,
    #     batch_norm=True, short_cut=wandb.config.target_short,
    #     concat_hidden=wandb.config.target_concat, readout=wandb.config.target_readout
    # )
    protein_model = models.GeoBind(
        input_dim=20, embedding_dim=wandb.config.target_hidden, num_layers=wandb.config.target_layer,
        radius=wandb.config.target_raidus  # topk=wandb.config.target_topk,
    )
    # ============================== Task Defining ============================== #
    task = tasks.NodePropertyPrediction(
        model=protein_model, metric=("macro_auroc", "macro_auprc"), num_mlp_layer=2
    )
    task.apply(initialize_weights)

    # ============================== Normal Lr setting ============================== #
    optimizer = torch.optim.AdamW(
        params=task.parameters(), lr=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay
    )
    solver = core.Engine(
        task, train_set, valid_set, test_set, optimizer, None, gpus=[0],
        batch_size=wandb.config.batch_size, logger="wandb"
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='max', factor=0.2, patience=6, min_lr=1e-5  # change the mode in early_stopping
    )
    whole_params = sum(p.numel() for p in task.parameters())
    print(f'#The Whole Params: {whole_params}')

    # ============================== Training Begin ============================== #
    early_stopping = core.EarlyStopping(patience=15, mode='max')
    checkpoint = "../../result/model_pth/PD_Sweep_PDBbind_R" + str(wandb.config.target_raidus) + ".pth"
    # print(checkpoint)
    # checkpoint = "../../result/model_pth/PD_Sweep_PDBbind.pth"

    for epoch in range(200):
        print(">>>>>>>>   Model' LR: ", optimizer.param_groups[0]['lr'])
        solver.train()
        metric = solver.evaluate("valid")['macro_auroc']
        scheduler.step(metrics=metric)
        early_stopping(val_loss=metric, solver=solver, path=checkpoint)
        if early_stopping.early_stop:
            print(">>>>>>>>   Early stopping   >>>>>>>>")
            break
    solver.load(checkpoint)
    solver.evaluate("test")


wandb.agent(sweep_id=sweep_id, function=main, count=4)
wandb.finish()

