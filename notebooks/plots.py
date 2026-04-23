# plot accuracies during training
import pickle
import os
import numpy as np
ROOT_DIR = "/scratch/workspace/ppruthi_umass_edu-MI/task_based_cg_debugging/task_based_compositional_generalization"

mean_acc_results = {}
sharp_acc_results = {}
function_type = "diverse"
pe = "rel_global"
mode = "direct"
nsamples = [1, 5, 10, 50, 100, 170, 200, 500, 1000]
for K in [2, 6]:
    sharp_acc_results[str(K)] = {}
    mean_acc_results[str(K)] = {}
    for ns in nsamples:
        
        dir_path = f"{ROOT_DIR}/models/eval/{function_type}/fixed/nalph_26_seqlen_6_fnlen_6_taskmaxlen_{K}/{mode}/train_combination_{K}/eval_combination_{K}/{pe}/nh6_nl3/seed_0/sample_efficiency/nsamples_{ns}"
        
        
        # dir_path = f"{ROOT_DIR}/models/eval/uniform/fixed/nalph_26_seqlen_6_fnlen_6_taskmaxlen_{K}/direct/train_combination_{K}/eval_combination_{K}/abs/nh12_nl12/seed_40"
        
        # get all files in the directory
        try:
            files = os.listdir(dir_path)
        except:
            continue
        sharp_acc_results[str(K)][ns] = {}
        mean_acc_results[str(K)][ns] = {}
        
        # skip accs.pkl
        files = [file for file in files if (file != 'accs_19601.pkl') or (file != 'representation_results_19601.pkl')]
        # remove accs_19601.pkl
        # files = [file for file in files if file != 'accs_19601.pkl']
        files = [file for file in files if file != 'representation_results_19601.pkl']
        # sort the files by the number in the file name
        files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        print(K, ns, files)
        for file in files:
            file_path = os.path.join(dir_path, file)
            with open(file_path, 'rb') as f:
                accs = pickle.load(f)
            i = int(file.split('_')[-1].split('.')[0])
            
            
            for split, res in accs.items():
                # if type of numpy array skip
                if isinstance(res, (np.ndarray, list)):
                    continue
                sharp_acc_results[str(K)][ns][split] = res.sharp_accuracy
                mean_acc_results[str(K)][ns][split] = res.mean_accuracy

import numpy as np
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(70, 15), sharey=True)
K_values = [2, 6]
# set font size to 20
font_size = 50
plt.rcParams.update({'font.size': font_size})
plt.rcParams.update({'legend.fontsize': font_size})
plt.rcParams.update({'axes.labelsize': font_size})
plt.rcParams.update({'axes.titlesize': font_size})
colors = {'train': 'blue', 'val': 'green', 'test': 'red'}

# To collect the handles/labels for a unified legend
compositions = {"2": 30, "6": 720}
handles_labels = []
markersize = 20
linewidth = 4
for idx, K in enumerate(K_values):
    ax = axes[idx]
    # Prepare iteration list (sorted for consistent plotting)
    iters = sorted(list(sharp_acc_results[str(K)].keys()))
    print(iters)
    # Gather accuracy lists
    
    print(str(K))
    train_sharp_accs = [sharp_acc_results[str(K)][iter]['train'] for iter in iters]
    val_sharp_accs = [sharp_acc_results[str(K)][iter]['train_heldout'] for iter in iters]
    test_sharp_accs = [sharp_acc_results[str(K)][iter]['test'] for iter in iters]
    train_mean_accs = [mean_acc_results[str(K)][iter]['train'] for iter in iters]
    val_mean_accs = [mean_acc_results[str(K)][iter]['train_heldout'] for iter in iters]
    test_mean_accs = [mean_acc_results[str(K)][iter]['test'] for iter in iters]
    

    l1, = ax.plot(iters, train_sharp_accs, label='Train sharp acc', color=colors['train'], linewidth=linewidth, marker='o', markersize=markersize)
    l2, = ax.plot(iters, train_mean_accs, label='Train mean acc', color=colors['train'], linestyle='--', linewidth=linewidth, marker='v', markersize=markersize, alpha=0.5)
    l3, = ax.plot(iters, val_sharp_accs, label='Val sharp acc', color=colors['val'], linewidth=linewidth, marker='o', markersize=markersize)
    l4, = ax.plot(iters, val_mean_accs, label='Val mean acc', color=colors['val'], linestyle='--', linewidth=linewidth, marker='v', markersize=markersize, alpha=0.5)
    l5, = ax.plot(iters, test_sharp_accs, label='Test sharp acc', color=colors['test'], linewidth=linewidth, marker='o', markersize=markersize)
    l6, = ax.plot(iters, test_mean_accs, label='Test mean acc', color=colors['test'], linestyle='--', linewidth=linewidth, marker='v', markersize=markersize, alpha=0.5)
    if idx == 0:
        handles_labels = [(l1, 'Train sharp acc'),
                          (l2, 'Train mean acc'),
                          (l3, 'Val sharp acc'),
                          (l4, 'Val mean acc'),
                          (l5, 'Test sharp acc'),
                          (l6, 'Test mean acc')]
    ax.set_title(f'K={K}')
    total_compositions = int(0.8*compositions[str(K)])
    ax.set_xlabel(f'Number of samples per composition (N = {total_compositions})', fontsize = font_size)
    # set xticks to iters
    ax.set_xticks(iters)
    # 45 degrees
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize = font_size)
    # set yticks to fontsize
    ax.set_yticklabels(ax.get_yticklabels(), fontsize = font_size)

    ax.grid(True)
    if idx == 0:
        ax.set_ylabel('Accuracy', fontsize = font_size)
    ax.set_ylim(-0.05, 1.05)  # give more room at the top
    # Do not include ax.legend() here
    for line in [l1, l2, l3, l4, l5, l6]:
        line.set_clip_on(False)
    ax.set_clip_on(False)
# One legend below all axes
handles, labels = zip(*handles_labels)
fig.legend(handles, labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.2))

# make plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)
plt.savefig(f"plots/sample_efficiency_curve_{function_type}_{mode}_nh6_nl3_pe_{pe}.pdf", dpi=300, bbox_inches='tight')

