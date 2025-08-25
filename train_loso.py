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
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import LeaveOneGroupOut
from utils.push_to_hf import push_model_to_hub
import torch.optim.lr_scheduler as lr_scheduler
from utils.messages import success, fail
from utils.pickle_patch import patch_pickle_loading
from preprocessing.transformations import DatasetReshape
from dataset_augmentations import AugmentedDatasetReshape, MixupAugmentedDataset
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
    INITIAL_LEARNING_RATE = 1e-4

    # Minimum learning rate
    MINIMUM_LEARNING_RATE = 1e-5

    # Weight decay for regularization
    WEIGHT_DECAY = 1e-3  # Increased from 1e-3 to 5e-3 to combat 77% overlap overfitting
    # Label smoothing for loss function
    LABEL_SMOOTHING = 0.20

    # Number of workers for data loading
    NUM_WORKERS = args.num_workers

    # % of data to randomly drop for regularization
    DATA_DROP_RATIO = 0.15

    # AUGMENTATION CONFIG - BDE-specific augmentations based on 1.4M sample analysis
    ENABLE_AUGMENTATIONS = True  # Enable augmentations to reduce overfitting
    AUGMENTATION_TYPE = "moderate"  # "conservative", "moderate", "aggressive"
    AUGMENTATION_PROBABILITY = 0.4  # What % of samples to augment (0.3=30%, 0.5=50%, etc.)
    ENABLE_MIXUP = True  # Keep disabled for now
    REPLACE_SMOTE_WITH_AUGMENTATIONS = True  # Keep SMOTE + augmentations

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
    EMBED_DIM = 256

    # Number of InterCorticalAttention Transformer Blocks
    DEPTH = 6

    # Number of parallel attention heads per InterCorticalAttention Transformer Block
    HEADS = 8

    # Dim of individual attention head in MHSA
    HEAD_DIM = 32

    # Hidden layer dimension of the Feed-Forward Network (FFN)
    MLP_HIDDEN_DIM = 256

    # Dropout Prob
    DROPOUT = 0.15

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

    # (Removed focal loss & EMA helpers; reverting to plain CrossEntropy)

    for fold, (train_idx, val_idx) in enumerate(logo.split(X_full, y_full, groups)):
        held_out_subject = groups[val_idx[0]]  # Which subject is held out
        print(f"\nFold {fold + 1}/{n_splits} - Testing on Subject {held_out_subject}")

        X_train = X_full[train_idx]
        y_train = y_full[train_idx]
        X_val = X_full[val_idx]
        y_val = y_full[val_idx]

        # Extract subject IDs for training set (for mixup)
        groups_train = groups[train_idx]

        # Data balancing: SMOTE vs Augmentations
        if REPLACE_SMOTE_WITH_AUGMENTATIONS:
            # Use augmentations instead of SMOTE
            print(f"Using augmentations instead of SMOTE for balancing")
            X_train_balanced, y_train_balanced = X_train, y_train
            groups_train_balanced = groups_train
        else:
            # Apply SMOTE to the training set (original approach)
            smote = SMOTE(random_state=SEED_VAL)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            
            # Properly extend subject IDs to match SMOTE output
            # SMOTE can create varying numbers per class, so we need to map back
            print(f"Original training size: {len(y_train)}, After SMOTE: {len(y_train_balanced)}")
            
            # Simple approach: assign each augmented sample the same subject ID as original
            # Find which original sample each SMOTE sample corresponds to
            groups_train_balanced = []
            for i in range(len(y_train_balanced)):
                # For SMOTE, we approximate by cycling through original subject IDs
                original_idx = i % len(groups_train)
                groups_train_balanced.append(groups_train[original_idx])
            groups_train_balanced = np.array(groups_train_balanced)

        # Create augmented datasets
        if ENABLE_AUGMENTATIONS:
            try:
                # Create custom config with controlled augmentation probability
                from augmentations import AUGMENTATION_CONFIGS
                custom_config = AUGMENTATION_CONFIGS[AUGMENTATION_TYPE].copy()
                custom_config['augmentation_probability'] = AUGMENTATION_PROBABILITY
                
                print(f"Using {AUGMENTATION_TYPE} augmentations on {AUGMENTATION_PROBABILITY*100:.0f}% of samples")
                
                if ENABLE_MIXUP:
                    print(f"Creating MixupAugmentedDataset with {AUGMENTATION_TYPE} augmentations")
                    # Add validation for dataset creation
                    print(f"Validation - X_train_balanced: {X_train_balanced.shape}")
                    print(f"Validation - y_train_balanced: {y_train_balanced.shape}")  
                    print(f"Validation - groups_train_balanced: {len(groups_train_balanced)}")
                    
                    train_dataset = MixupAugmentedDataset(
                        X_train_balanced, 
                        y_train_balanced,
                        groups_train_balanced,
                        num_electrodes=NUM_ELECTRODES,
                        apply_augmentations=True,
                        augmentation_config=custom_config,  # Use custom config
                        enable_mixup=True,
                        seed=SEED_VAL
                    )
                else:
                    print(f"Creating AugmentedDatasetReshape with {AUGMENTATION_TYPE} augmentations")
                    train_dataset = AugmentedDatasetReshape(
                        X_train_balanced, 
                        y_train_balanced, 
                        num_electrodes=NUM_ELECTRODES,
                        apply_augmentations=True,
                        augmentation_config=custom_config,  # Use custom config
                        subject_ids=groups_train_balanced,  # Pass subject IDs for group-wise augmentation
                        seed=SEED_VAL
                    )
            except Exception as e:
                print(f"Error creating augmented dataset: {e}")
                print("Falling back to standard DatasetReshape")
                train_dataset = DatasetReshape(
                    X_train_balanced, y_train_balanced, NUM_ELECTRODES
                )
        else:
            # Original dataset without augmentations
            print("Creating standard DatasetReshape (no augmentations)")
            train_dataset = DatasetReshape(
                X_train_balanced, y_train_balanced, NUM_ELECTRODES
            )

        # Validation dataset (always without augmentations)
        val_dataset = DatasetReshape(X_val, y_val, NUM_ELECTRODES)
        
        # Print final dataset sizes
        print(f"Final train dataset size: {len(train_dataset)}")
        print(f"Final val dataset size: {len(val_dataset)}")

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
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=INITIAL_LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        # Replace CosineAnnealingLR with ReduceLROnPlateau for better stability with overlapping windows
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=15, min_lr=MINIMUM_LEARNING_RATE
        )

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
                "scheduler": "ReduceLROnPlateau",
                "augmentations_enabled": ENABLE_AUGMENTATIONS,
                "augmentation_type": AUGMENTATION_TYPE if ENABLE_AUGMENTATIONS else None,
                "mixup_enabled": ENABLE_MIXUP if ENABLE_AUGMENTATIONS else False,
                "smote_replaced_by_augmentations": REPLACE_SMOTE_WITH_AUGMENTATIONS,
            },
        )

        for epoch in range(NUM_EPOCHS):
            # Use consistent batch size throughout training (removed dynamic batch size change)
            current_batch_size = INITIAL_BATCH_SIZE

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
                
                # Add gradient clipping to prevent exploding gradients from 77% window overlap
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()

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

            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)

                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(y.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            # Update scheduler based on validation loss (ReduceLROnPlateau needs validation loss)
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
                    "dropout": DROPOUT,
                    "weight_decay": WEIGHT_DECAY,
                    "loss_type": "CrossEntropyLoss",
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
            # (Removed focal/EMA logging)            
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
