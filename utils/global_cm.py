import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ SETTINGS ------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "train")

DATASETS = ["EMODB", "EMOVO", "RAVDESS"]
SPLITS = {
    "gender": "gender",   # 80-20 split
    "speaker": "speaker"  # LOO split
}

CONDITIONS = ["both", "gendered"]  # 'both' = no gender recognition, 'gendered' = male+female averaged

# Labels per dataset
LABELS_DICT = {
    "EMODB": ['angry', 'bored', 'neutral', 'disgust', 'fear', 'happy', 'sad'],

    "EMOVO": [
        'disgust\n(dis)', 
        'joy\n(gio)', 
        'neutral\n(neu)',
        'fear\n(pau)', 
        'angry\n(rab)', 
        'surprise\n(sor)', 
        'sad\n(tri)'
    ],

    "RAVDESS": ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
}

DISPLAY_NAMES = {
    "gender": "80–20",
    "speaker": "LOO"
}

# ------------------ LOAD & ORGANIZE MATRICES ------------------
# Structure: global_cms[dataset][split][condition] = list of normalized matrices
global_cms = {ds: {split: {cond: [] for cond in CONDITIONS} for split in SPLITS.keys()} for ds in DATASETS}

for ds in DATASETS:
    ds_path = os.path.join(BASE_DIR, ds)
    
    for split_name, split_key in SPLITS.items():
        # Temporary storage for male + female
        male_female_matrices = []

        for fname in os.listdir(ds_path):
            if not fname.endswith(".npy") or "_193" in fname:
                continue

            fullpath = os.path.join(ds_path, fname)
            arr = np.load(fullpath)

            # Ensure shape (#folds, C, C)
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=0)

            # Normalize to row-wise percentages
            arr_percent = arr / arr.sum(axis=2, keepdims=True) * 100

            # Assign to condition
            if "both" in fname:
                global_cms[ds][split_name]["both"].append(arr_percent)
            elif "male" in fname or "female" in fname:
                male_female_matrices.append(arr_percent)

        # Stack 'both' matrices
        if len(global_cms[ds][split_name]["both"]) > 0:
            global_cms[ds][split_name]["both"] = np.concatenate(global_cms[ds][split_name]["both"], axis=0)
        else:
            global_cms[ds][split_name]["both"] = None

        # Average male+female matrices per split, handling different #folds
        if len(male_female_matrices) > 0:
            # Concatenate along a new first axis
            all_matrices = []
            for arr in male_female_matrices:
                all_matrices.append(arr)  # arr.shape = (#folds, C, C)

            # Now we need to pad/truncate to the same number of folds if desired,
            # or just concatenate all folds together
            all_folds = np.concatenate(all_matrices, axis=0)  # shape = (sum_of_folds, C, C)
            global_cms[ds][split_name]["gendered"] = all_folds
        
        else:
            global_cms[ds][split_name]["gendered"] = None



# ------------------ PLOTTING ------------------
fig, axs = plt.subplots(len(DATASETS), len(SPLITS)*len(CONDITIONS), figsize=(60, 40))

for r, ds in enumerate(DATASETS):
    labels = LABELS_DICT[ds]
    for c_s, split_name in enumerate(SPLITS.keys()):
        for c_cond, cond_name in enumerate(CONDITIONS):
            ax = axs[r, c_s*len(CONDITIONS) + c_cond]
            cm = global_cms[ds][split_name][cond_name]

            if cm is None:
                ax.set_visible(False)
                continue

            mean_cm = np.mean(cm, axis=0)
            std_cm = np.std(cm, axis=0)

            # Build annotations
            cell_labels = np.empty(mean_cm.shape, dtype=object)
            for i in range(mean_cm.shape[0]):
                for j in range(mean_cm.shape[1]):
                    cell_labels[i, j] = f"{mean_cm[i, j]:.1f}±{std_cm[i, j]:.1f}"

            # Heatmap
            sns.heatmap(mean_cm, annot=cell_labels, fmt="", cmap="viridis",
                        ax=ax, cbar=(r==0 and c_s==1 and c_cond==1), annot_kws={"size": 7}, vmin=0, vmax=100)

            ax.set_xticks(np.arange(len(labels)) + 0.5)
            ax.set_yticks(np.arange(len(labels)) + 0.5)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels, rotation=0)

            ax.set_title(f"{ds} {DISPLAY_NAMES[split_name]}: {'No Gender' if cond_name=='both' else 'With Gender'}")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")

plt.subplots_adjust(hspace=0.6, wspace=0.4)
plt.show()
