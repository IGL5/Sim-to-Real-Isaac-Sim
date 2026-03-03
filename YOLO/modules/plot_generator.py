import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(tp, fp, fn, output_path):
    """ Draws a confusion matrix """
    plt.figure(figsize=(6, 5))
    matrix = [[tp, fp], [fn, 0]]
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, 
                xticklabels=["Pred Pos", "Pred Neg"], yticklabels=["Real Pos", "Real Neg"])
    plt.title("Confusion Matrix")
    plt.savefig(output_path)
    plt.close()

def plot_confidence_histogram(confs_primary, label_primary, color_primary, output_path, title, confs_secondary=None, label_secondary=None, color_secondary=None):
    """ Draws a confidence histogram """
    plt.figure(figsize=(8, 5))
    bins_fixed = np.linspace(0, 1, 21)
    plt.hist(confs_primary, bins=bins_fixed, alpha=0.7, label=label_primary, color=color_primary)
    
    if confs_secondary is not None:
        plt.hist(confs_secondary, bins=bins_fixed, alpha=0.7, label=label_secondary, color=color_secondary)
        plt.legend()
        
    plt.title(title)
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.xlim([0.0, 1.0])
    plt.savefig(output_path)
    plt.close()

def plot_normalized_heatmap(centers, output_path, title="Normalized Detection Heatmap", cmap='inferno'):
    """ Draws a normalized detection heatmap """
    if not centers:
        return
        
    c_arr = np.array(centers)
    plt.figure(figsize=(8, 6))
    plt.hexbin(c_arr[:, 0], c_arr[:, 1], gridsize=20, cmap=cmap, mincnt=1, extent=[0, 1, 0, 1])
    plt.colorbar(label='Detections')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.xlabel("Normalized Width (0.0 - 1.0)")
    plt.ylabel("Normalized Height (0.0 - 1.0)")
    plt.savefig(output_path)
    plt.close()

def plot_pr_curve(precisions, recalls, ap50, output_path):
    """ Draws a Precision-Recall curve """
    plt.figure(figsize=(8, 5))
    plt.plot(recalls, precisions, color='blue', lw=2, label=f'PR Curve (mAP@50 = {ap50:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (IoU=0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_path)
    plt.close()

def plot_comparison_bar(labels, data_lists, data_labels, output_path, title):
    """ Draws a grouped bar chart comparing several models """
    x = np.arange(len(labels))
    width = 0.8 / len(data_lists)
    
    fig, ax = plt.subplots(figsize=(max(8, len(labels)*2.5), 6))
    
    for i, data in enumerate(data_lists):
        offset = (i - len(data_lists)/2 + 0.5) * width
        ax.bar(x + offset, data, width, label=data_labels[i], alpha=0.8)
        
    ax.set_ylabel('Valor (0.0 - 1.0)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_simple_bar(labels, data, ylabel, output_path, title):
    """ Draws a simple bar chart """
    plt.figure(figsize=(max(6, len(labels)*2), 5))
    sns.barplot(x=labels, y=data, palette="viridis")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()