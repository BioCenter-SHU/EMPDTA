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


# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
warnings.filterwarnings("ignore")

sweep_configuration = {
    'method': 'grid',
    'parameters':
    {
        'random_seed': {'values': [42]},
        'num_epoch': {'values': [200]},
        'batch_size': {'values': [32]},
        'fold': {'values': [0, 1, 2, 3, 4]},  # 0, 1, 2, 3, 4
        'learning_rate': {'values': [1e-4]},  # 1e-3
        'weight_decay': {'values': [1e-4]},
        'target_readout': {'values': ["mean"]},
        'target_short': {'values': [True]},
        'target_concat': {'values': [True]},
        'target_hidden': {'values': [512]},
        'target_layer': {'values': [4]},
        'drug_readout': {'values': ["mean"]},
        'drug_short': {'values': [False]},
        'drug_concat': {'values': [False]},
        'drug_hidden': {'values': [256]},
        'drug_layer': {'values': [5]},
        'point_embedding': {'values': [16]},
        'point_layer': {'values': [3]},
        'point_topk': {'values': [256]},  # 32, 64, 128, 256
        'point_radius': {'values': [15.0]},  # 6.0, 9.0, 12.0, 15.0
        # 'loss_weight': {'values': [0.5]}
    }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project='AIO Filter Davis Sweep')
# sweep_id = "marinehdk/AIO Filter Davis Sweep/hvr753tk"


def main():
    wandb.init()
    seed_torch(wandb.config.random_seed)
    # ============================== Dataset Loading ============================== #
    transform = transforms.ProteinView(view="residue", keys="graph1")
    dataset = datasets.FilteredDavis(protein_method="gearnetesm650m", drug_method="distanceMol", transform=transform)

    train_set, valid_set, test_set = dataset.deepdta_split(fold=wandb.config.fold)
    # train_set, valid_set, test_set = dataset.ccv_split(fold=wandb.config.fold)
    print(f"Train samples: {len(train_set)}, Valid samples: {len(valid_set)}, Test samples: {len(test_set)}")
    # ============================== Model Defining  ============================== #
    target_model = models.GearNet(
        input_dim=1280, hidden_dims=[wandb.config.target_hidden] * wandb.config.target_layer,
        num_relation=7, edge_input_dim=59, batch_norm=True, short_cut=wandb.config.target_short,
        concat_hidden=wandb.config.target_concat, readout=wandb.config.target_readout
    )

    drug_model = models.RGCN(
        input_dim=67, hidden_dims=[wandb.config.drug_hidden] * wandb.config.drug_layer,
        num_relation=4, edge_input_dim=19, batch_norm=True, short_cut=wandb.config.drug_short,
        concat_hidden=wandb.config.drug_concat, readout=wandb.config.drug_readout
    )

    point_model = models.GeoBind(
        input_dim=20, embedding_dim=wandb.config.point_embedding, num_layers=wandb.config.point_layer,
        topk=wandb.config.point_topk, radius=wandb.config.point_radius
    )

    # ============================== Task Defining ============================== #
    task = tasks.AffinityWithDocking(
        target_model=target_model, drug_model=drug_model, point_model=point_model, mode_type="MolFormer",
        task=dataset.tasks, criterion_weight={"Affinity MSE": 1},  # , "Pocket BCE": wandb.config.loss_weight
        criterion="mse", metric=("rmse", "mse", "c_index", "pearsonr", "spearmanr", "rm2", "macro_auroc", "macro_auprc")
    )
    task.apply(initialize_weights)

    # ============================== Pretrained params loading for both models ============================== #
    # model_checkpoint = "../../result/model_pth/POD_Pretrain_FD.pth"
    # model_checkpoint = torch.load(model_checkpoint)["model"]
    #
    # target_checkpoint = {}
    # for k, v in model_checkpoint.items():
    #     if k.startswith("model"):
    #         target_checkpoint[k[6:]] = v
    # task.point_model.load_state_dict(target_checkpoint, strict=False)

    # ============================== Fix the params except MLP ============================== #
    # for name, param in task.point_model.named_parameters():
    #     # fix all params except the atom_net pocket mlp prediction
    #     if "conv_layers" in name:
    #         param.requires_grad = True
    #     elif "out_layers" in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False
    #
    # print("============================== Fix the POR MLP params ==============================")

    # ============================== Normal Lr setting ============================== #
    optimizer = torch.optim.AdamW(
        params=task.parameters(), lr=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.2)
    solver = core.Engine(
        task, train_set, valid_set, test_set, optimizer, None, gpus=[0],
        batch_size=wandb.config.batch_size, logger="wandb"
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='min', factor=0.2, patience=6, min_lr=1e-5
    )
    whole_params = sum(p.numel() for p in task.parameters())
    print(f'#The Whole Params: {whole_params}')

    # ============================== Training Begin ============================== #
    early_stopping = core.EarlyStopping(patience=15)
    checkpoint = "../../result/model_pth/MD_AIO_Sweep_Davis.pth"

    for epoch in range(wandb.config.num_epoch):
        print(">>>>>>>>   Model' LR: ", optimizer.param_groups[0]['lr'])
        solver.train()
        metric = solver.evaluate("valid")['root mean squared error [affinity]']
        scheduler.step(metrics=metric)
        early_stopping(val_loss=metric, solver=solver, path=checkpoint)
        if early_stopping.early_stop:
            print(">>>>>>>>   Early stopping   >>>>>>>>")
            break
    solver.load(checkpoint)
    solver.evaluate("test")


wandb.agent(sweep_id=sweep_id, function=main, count=5)
wandb.finish()
