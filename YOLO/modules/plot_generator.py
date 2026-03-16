import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(tp, fp, fn, output_path):
    """ Draws a confusion matrix """
    tn = 0 
    matrix = [[tp, fn], 
              [fp, tn]]

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                     xticklabels=['Pred Pos', 'Pred Neg'],
                     yticklabels=['Real Pos', 'Real Neg'])
    plt.title('Confusion Matrix')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90, va="center")
    plt.tight_layout()
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

def plot_f1_curve(confs, f1_scores, best_conf, best_f1, output_path):
    """ Draws an F1-Score vs Confidence curve to find the optimal threshold """
    plt.figure(figsize=(8, 5))
    plt.plot(confs, f1_scores, color='green', lw=2, label=f'F1 Curve (Peak: {best_f1:.2f} @ {best_conf:.3f})')
    plt.scatter([best_conf], [best_f1], color='red', zorder=5)
    plt.axvline(x=best_conf, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Confidence Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 vs Confidence (Threshold Optimization)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower center")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_path)
    plt.close()