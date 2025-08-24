import os
import torch
import wandb
import pickle
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
# from collections import Counter
from utils.seed import set_seed
from huggingface_hub import login
from model.model import RBTransformer
from torch.utils.data import DataLoader
from sklearn.model_selection import LeaveOneGroupOut
from utils.push_to_hf import push_model_to_hub
import torch.optim.lr_scheduler as lr_scheduler
from utils.messages import success, fail
from utils.pickle_patch import patch_pickle_loading
from preprocessing.transformations import DatasetReshape
from utils.auto_detect import get_num_electrodes, get_num_classes
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

################################################################################
# ARGPARSE CONFIGURATION
################################################################################


def parse_args():
    parser = argparse.ArgumentParser(description="RBTransformer EEG Training Script")
    parser.add_argument("--root_dir", type=str, default="preprocessed_datasets")
    parser.add_argument(
        "--dataset_name", type=str, required=True, choices=["seed", "deap", "dreamer"]
    )
    parser.add_argument(
        "--task_type", type=str, required=True, choices=["binary", "multi"]
    )
    parser.add_argument("--dimension", type=str, required=True)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--hf_username", type=str, required=True)
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--wandb_api_key", type=str, required=True)
    return parser.parse_args()


################################################################################
# MAIN-FUNCTION
################################################################################
def main():
    args = parse_args()

    ################################################################################
    # PREPROCESSED-DATASETS
    ################################################################################
    PREPROCESSED_DATASETS = {
        "deap": {
            "binary": {
                "valence": "deap_binary_valence_dataset.pkl",
                "arousal": "deap_binary_arousal_dataset.pkl",
                "dominance": "deap_binary_dominance_dataset.pkl",
            },
            "multi": {
                "valence": "deap_multi_valence_dataset.pkl",
                "arousal": "deap_multi_arousal_dataset.pkl",
                "dominance": "deap_multi_dominance_dataset.pkl",
            },
        },
        "dreamer": {
            "binary": {
                "valence": "dreamer_binary_valence_dataset.pkl",
                "arousal": "dreamer_binary_arousal_dataset.pkl",
                "dominance": "dreamer_binary_dominance_dataset.pkl",
            },
            "multi": {
                "valence": "dreamer_multi_valence_dataset.pkl",
                "arousal": "dreamer_multi_arousal_dataset.pkl",
                "dominance": "dreamer_multi_dominance_dataset.pkl",
            },
        },
        "seed": {
            "multi": {
                "emotion": "seed_multi_emotion_dataset.pkl",
            },
        },
    }

    # Select Preprocessed Dataset
    DATASET_NAME = args.dataset_name  # Options: "seed", "deap", "dreamer"
    CLASSIFICATION_TYPE = (
        args.task_type
    )  # Options: "multi" For SEED; "binary", "multi" For DEAP/DREAMER
    DIMENSION = (
        args.dimension
    )  # Options: "emotion" For SEED, "valence", "arousal", "dominance" For DEAP/DREAMER
    dataset_path = os.path.join(
        args.root_dir,
        PREPROCESSED_DATASETS[DATASET_NAME][CLASSIFICATION_TYPE][DIMENSION],
    )

    print(
        success(
            f"Training Initialized => Dataset: {DATASET_NAME.upper()} || Dimension: {DIMENSION.capitalize()} || Task: {CLASSIFICATION_TYPE.capitalize()} Class Classification"
        )
    )

    # Load Preprocessed Dataset
    patch_pickle_loading()

    try:
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)
        print(success(f"Success: Dataset '{dataset_path}' successfully loaded"))
    except Exception as e:
        print(fail(f"Failed: Dataset '{dataset_path}' failed to load"))
        raise e

    ################################################################################
    # SEED CONFIG
    ################################################################################
    SEED_VAL = args.seed
    set_seed(SEED_VAL)
    print(success(f"Seed value set for training run: {SEED_VAL}"))

    ################################################################################
    # TRAINING-HYPERPARAMETERS
    ################################################################################
    # Number of training epochs
    NUM_EPOCHS = 300

    # (Removed K-Fold; using pure LOSO)

    # Initial batch size, First half of training
    INITIAL_BATCH_SIZE = 256

    # Reduced batch size, Second half of training
    REDUCED_BATCH_SIZE = 64

    # Starting learning rate
    INITIAL_LEARNING_RATE = 1e-3

    # Minimum learning rate
    MINIMUM_LEARNING_RATE = 1e-6

    # Weight decay for regularization
    # Increased weight decay for stronger regularization
    WEIGHT_DECAY = 5e-3

    # Label smoothing for loss function
    LABEL_SMOOTHING = 0.0  # retained (not used in focal loss)

    # Number of workers for data loading
    NUM_WORKERS = args.num_workers

    # % of data to randomly drop for regularization
    DATA_DROP_RATIO = 0.10

    ################################################################################
    # MODEL-CONFIG
    ################################################################################
    # Number of electrodes in datasets (SEED=62, DEAP=32, DREAMER=14)
    NUM_ELECTRODES = get_num_electrodes(args.dataset_name)

    # Number of logit classes (SEED=3, DEAP-binary=2, DEAP-multi=9, DREAMER-binary=2, DREAMER-multi=5)
    NUM_CLASSES = get_num_classes(args.dataset_name, args.task_type)

    # BDE Tokens dim
    BDE_DIM = 4

    # Projected dimension of BDE tokens
    EMBED_DIM = 128

    # Number of InterCorticalAttention Transformer Blocks
    DEPTH = 4

    # Number of parallel attention heads per InterCorticalAttention Transformer Block
    HEADS = 6

    # Dim of individual attention head in MHSA
    HEAD_DIM = 32

    # Hidden layer dimension of the Feed-Forward Network (FFN)
    MLP_HIDDEN_DIM = 128

    # Dropout Prob
    # Increased dropout
    DROPOUT = 0.3

    # Focal loss gamma
    FOCAL_GAMMA = 2.0

    # EMA decay
    EMA_DECAY = 0.999

    # Gradient clipping max norm
    GRAD_CLIP_MAX_NORM = 1.0

    # Device Set (GPU if avail else CPU)
    DEVICE = torch.device(args.device)
    print(success(f"Device Set: {DEVICE}"))

    ################################################################################
    #  WANDB & HUGGINGFACE CONFIG
    ################################################################################
    # WANDB: Key and Login
    try:
        wandb.login(key=args.wandb_api_key)
        print(success("Success: WandB API key authenticated."))
    except Exception as e:
        print(fail("Failed: WandB API key authentication failed."))
        raise e

    # WANDB RUN COMPS
    PAPER_TASK = "α-rbtransformer-eeg-recognition"
    TASK_TYPE_SUFFIX = "class-classification"
    RUN_TAG = "loso-sota-run"
    RUN_ID = "0001"
    WANDB_RUN_NAME = f"{PAPER_TASK}-{DATASET_NAME}-{CLASSIFICATION_TYPE}-{DIMENSION}-{TASK_TYPE_SUFFIX}-{RUN_TAG}-{RUN_ID}"

    # HF: Key and Login
    try:
        login(token=args.hf_token)
        print(success("Success: Hugging Face token authenticated."))
    except Exception as e:
        print(fail("Failed: Hugging Face token authentication failed."))
        raise e

    # HF MODEL REPO_ID
    USERNAME = args.hf_username
    BASE_REPO_ID = f"{USERNAME}/{DATASET_NAME}-{CLASSIFICATION_TYPE}-{DIMENSION}-LOSO"

    ################################################################################
    # REGULARIZATION: DATA DROPOUT
    ################################################################################
    X_full = []
    y_full = []
    groups = []  # Add groups for LOSO
    for i in range(len(dataset)):
        x, y = dataset[i]
        subject_id = dataset._info_memory[i]['subject_id']  # Extract subject ID
        
        X_full.append(x.squeeze(0).numpy().flatten())
        y_full.append(y)
        groups.append(subject_id)  # Store subject ID for grouping

    X_full = np.array(X_full)
    y_full = np.array(y_full)
    groups = np.array(groups)  # Convert groups to numpy array

    # Data dropout is disabled for LOSO to keep full data collection
    # num_samples = len(X_full)
    # drop_count = int(num_samples * DATA_DROP_RATIO)
    # all_indices = np.arange(num_samples)

    # np.random.shuffle(all_indices)
    # kept_indices = all_indices[drop_count:]
    # X_full = X_full[kept_indices]
    # y_full = y_full[kept_indices]

    ################################################################################
    # LOSO (LEAVE-ONE-SUBJECT-OUT) TRAINING
    ################################################################################
    # Comment out old K-Fold approach
    # kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=SEED_VAL)
    
    # New LOSO Cross-Validation
    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(X_full, y_full, groups)
    
    print(success(f"LOSO Cross-Validation: {n_splits} subjects, {len(np.unique(groups))} unique subjects"))

    # ------------------------------
    # Helper: Focal Loss
    # ------------------------------
    class FocalLoss(nn.Module):
        def __init__(self, weight=None, gamma: float = 2.0, reduction: str = "mean"):
            super().__init__()
            self.weight = weight
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, logits, targets):
            ce = torch.nn.functional.cross_entropy(
                logits, targets, weight=self.weight, reduction="none"
            )
            pt = torch.exp(-ce)
            focal = (1 - pt) ** self.gamma * ce
            if self.reduction == "mean":
                return focal.mean()
            elif self.reduction == "sum":
                return focal.sum()
            return focal

    # ------------------------------
    # Helper: Exponential Moving Average (EMA) Wrapper
    # ------------------------------
    class ModelEMA:
        def __init__(self, model: nn.Module, decay: float = 0.999):
            self.ema_model = self._clone(model)
            self.decay = decay
            self.num_updates = 0

        def _clone(self, model):
            ema_model = type(model)(
                num_electrodes=model.num_electrodes,
                bde_dim=model.bde_dim,
                embed_dim=model.embed_dim,
                depth=model.depth,
                heads=model.heads,
                head_dim=model.head_dim,
                mlp_hidden_dim=model.mlp_hidden_dim,
                dropout=model.dropout,
                num_classes=model.num_classes,
            )
            ema_model.load_state_dict(model.state_dict())
            for p in ema_model.parameters():
                p.requires_grad_(False)
            ema_model.to(model.device)
            return ema_model

        @torch.no_grad()
        def update(self, model):
            self.num_updates += 1
            d = self.decay
            msd = model.state_dict()
            for k, v in self.ema_model.state_dict().items():
                if v.dtype.is_floating_point:
                    v.copy_(v * d + (1.0 - d) * msd[k])

    for fold, (train_idx, val_idx) in enumerate(logo.split(X_full, y_full, groups)):
        held_out_subject = groups[val_idx[0]]  # Which subject is held out
        print(f"\nFold {fold + 1}/{n_splits} - Testing on Subject {held_out_subject}")

        X_train = X_full[train_idx]
        y_train = y_full[train_idx]
        X_val = X_full[val_idx]
        y_val = y_full[val_idx]

        # Class weights (inverse frequency) for focal loss
        unique_classes, counts = np.unique(y_train, return_counts=True)
        class_weight_arr = np.zeros(NUM_CLASSES, dtype=np.float32)
        for cls, cnt in zip(unique_classes, counts):
            class_weight_arr[int(cls)] = len(y_train) / (NUM_CLASSES * cnt)
        class_weight_tensor = torch.tensor(class_weight_arr, dtype=torch.float32).to(DEVICE)

        train_dataset = DatasetReshape(X_train, y_train, NUM_ELECTRODES)
        val_dataset = DatasetReshape(X_val, y_val, NUM_ELECTRODES)

        # balanced_counts = Counter(y_train_balanced)
        # print(f"\nFold {fold + 1} Training Set Class Balance (After SMOTE):")
        # print(f"Total samples: {len(y_train_balanced)}")
        # for cls, count in balanced_counts.items():
        #     print(
        #         f"Class {cls}: {count} samples ({count / len(y_train_balanced) * 100:.2f}%)"
        #     )

        val_loader = DataLoader(
            val_dataset,
            batch_size=INITIAL_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

        # Init RBTransformer and moving it to the device
        model = RBTransformer(
            num_electrodes=NUM_ELECTRODES,
            bde_dim=BDE_DIM,
            embed_dim=EMBED_DIM,
            depth=DEPTH,
            heads=HEADS,
            head_dim=HEAD_DIM,
            mlp_hidden_dim=MLP_HIDDEN_DIM,
            dropout=DROPOUT,
            num_classes=NUM_CLASSES,
        )
        model = model.to(DEVICE)

        # loss, optimizer, and scheduler
        # Focal loss with class weighting
        criterion = FocalLoss(weight=class_weight_tensor, gamma=FOCAL_GAMMA)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=INITIAL_LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        # Scheduler switched to ReduceLROnPlateau (monitor val_loss)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=MINIMUM_LEARNING_RATE,
            verbose=False,
        )

        # Initialize EMA
        ema_helper = ModelEMA(model, decay=EMA_DECAY)

        # Init WandB for current fold
        wandb.init(
            project=WANDB_RUN_NAME,
            group=f"{DATASET_NAME}-{CLASSIFICATION_TYPE}-{DIMENSION}",
            name=f"LOSO-HELDOUT-SUBJECT-{held_out_subject}",
            config={
                "learning_rate": INITIAL_LEARNING_RATE,
                "minimum_learning_rate": MINIMUM_LEARNING_RATE,
                "num_epochs": NUM_EPOCHS,
                "model": "RBTransformer",
                "cv_method": "LOSO",
                "num_subjects": n_splits,
                "held_out_subject": held_out_subject,
                "fold": fold + 1,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
            },
        )

        for epoch in range(NUM_EPOCHS):
            if epoch < 150:
                current_batch_size = INITIAL_BATCH_SIZE
            else:
                current_batch_size = REDUCED_BATCH_SIZE

            train_loader = DataLoader(
                train_dataset,
                batch_size=current_batch_size,
                shuffle=True,
                num_workers=NUM_WORKERS,
            )

            # Training loop
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch in tqdm(
                train_loader,
                desc=f"LOSO-HELDOUT-SUBJECT-{held_out_subject} Epoch {epoch + 1}/{NUM_EPOCHS} - Training",
            ):
                x, y = batch
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
                optimizer.step()
                # EMA update
                ema_helper.update(model)

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += y.size(0)
                train_correct += (predicted == y).sum().item()

            train_accuracy = train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)

            # Eval loop
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []

            # Use EMA weights for validation inference
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    outputs = ema_helper.ema_model(x)
                    loss = criterion(outputs, y)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)

                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(y.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)

            # Scheduler step with monitored metric
            scheduler.step(avg_val_loss)

            # Calculate eval metrics
            val_accuracy = accuracy_score(all_targets, all_preds)
            precision = precision_score(
                all_targets, all_preds, average="macro", zero_division=0
            )
            recall = recall_score(
                all_targets, all_preds, average="macro", zero_division=0
            )
            f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

            # Log config and metrics to WandB
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": avg_val_loss,
                    "val_accuracy": val_accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_batch_size": current_batch_size,
                    "class_weights": class_weight_arr.tolist(),
                    "scheduler_metric": avg_val_loss,
                    "ema_decay": EMA_DECAY,
                    "focal_gamma": FOCAL_GAMMA,
                    "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
                    "dropout": DROPOUT,
                    "weight_decay": WEIGHT_DECAY,
                    "loss_type": "FocalLoss",
                }
            )

            # Log config and metrics to terminal
            tqdm.write(f"\n[LOSO-HELDOUT-SUBJECT-{held_out_subject}] Epoch {epoch + 1}/{NUM_EPOCHS}")
            tqdm.write(f"Train Loss     : {avg_train_loss:.4f}")
            tqdm.write(f"Train Accuracy : {train_accuracy:.4f}")
            tqdm.write(f"Val Loss       : {avg_val_loss:.4f}")
            tqdm.write(f"Val Accuracy   : {val_accuracy:.4f}")
            tqdm.write(f"Precision      : {precision:.4f}")
            tqdm.write(f"Recall         : {recall:.4f}")
            tqdm.write(f"F1 Score       : {f1:.4f}")
            tqdm.write(f"Learning Rate  : {optimizer.param_groups[0]['lr']:.6f}")
            tqdm.write(f"LR Scheduler Metric (val_loss): {avg_val_loss:.4f}")
            tqdm.write(f"Current Class Weights: {np.round(class_weight_arr, 3)}")
            tqdm.write(f"Batch Size     : {current_batch_size}")

        # Push model to Hub
        push_model_to_hub(
            model=model,
            repo_id=f"{BASE_REPO_ID}-Subject-{held_out_subject}",
            commit_message=f"Upload of trained RBTransformer on LOSO Subject-{held_out_subject} run",
        )

        wandb.finish()


if __name__ == "__main__":
    main()
