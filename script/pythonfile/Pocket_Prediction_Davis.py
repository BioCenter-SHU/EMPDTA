import os
import random
import torch
import warnings
import wandb
import numpy as np
from torchdrug import transforms, datasets, models, tasks, core


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
        # torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    # BatchNorm1d init
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


seed_torch()
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
warnings.filterwarnings("ignore")

# ============================== Dataset Loading ============================== #
transform = transforms.ProteinView(view="residue", keys="graph1")
dataset = datasets.DavisProtein(protein_method="gearnet", transform=transform)

train_set, valid_set, test_set = dataset.ccv_split()
print(f"Train samples: {len(train_set)}, Valid samples: {len(valid_set)}, Test samples: {len(test_set)}")

# ============================== Model Define ============================== #
# protein_model = models.GearNet(
#     input_dim=21, hidden_dims=[512, 512, 512], num_relation=7, short_cut=False,
#     batch_norm=True, concat_hidden=False, readout='mean'
# )
# protein_model = models.Conv1D(input_dim=21, hidden_dims=[32, 64, 96], kernel_size=7, padding=3)
protein_model = models.GeoBind(
        input_dim=20, embedding_dim=16, num_layers=5, threshold=0.9, radius=12.0
    )
task = tasks.NodePropertyPrediction(
    model=protein_model, metric=("macro_auroc", "macro_auprc"), num_mlp_layer=2
)
task.apply(initialize_weights)

# ============================== Normal Lr setting ============================== #
# wandb.init(project="Davis Pocket Prediction")
learning_rate = 1e-3
weight_decay = 1e-4
optimizer = torch.optim.AdamW(params=task.parameters(), lr=learning_rate, weight_decay=weight_decay)

solver = core.Engine(
    task, train_set, valid_set, test_set, optimizer, None, gpus=[0], batch_size=8  # , logger="wandb"
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, mode='max', factor=0.2, patience=10, min_lr=1e-5  # change the mode in early_stopping
)

whole_params = sum(p.numel() for p in task.parameters())
print(f'#The Params: {whole_params}')

# ============================== Training Begin ============================== #
early_stopping = core.EarlyStopping(patience=30, mode='max')
checkpoint = "../../result/model_pth/PocketPrediction_single.pth"

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
