import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(confusion_pairs, class_names, output_path):
    """ Draws a dynamic Multi-Class confusion matrix including Background """
    if not confusion_pairs:
        return

    # Get all unique classes involved, including -1 (Background)
    unique_classes = set([c for pair in confusion_pairs for c in pair])
    
    # Sort classes (0, 1, 2...) and leave -1 at the end
    sorted_classes = sorted([c for c in unique_classes if c != -1])
    if -1 in unique_classes:
        sorted_classes.append(-1)
    
    labels = [class_names.get(c, f"Class {c}") if c != -1 else "Background (BG)" for c in sorted_classes]
    
    # Create empty NxN matrix
    size = len(sorted_classes)
    matrix = np.zeros((size, size), dtype=int)
    
    # Map class ID to matrix row/column
    idx_map = {c: i for i, c in enumerate(sorted_classes)}
    
    # Fill the matrix counting the pairs (Real, Predicted)
    for real_c, pred_c in confusion_pairs:
        matrix[idx_map[real_c], idx_map[pred_c]] += 1
        
    # Draw
    plt.figure(figsize=(max(6, size*1.2), max(5, size*1.2)))
    ax = sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                     xticklabels=labels, yticklabels=labels)
    plt.title('Multi-Class Confusion Matrix')
    plt.ylabel('Real Label')
    plt.xlabel('YOLO Prediction')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90, va="center")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_confidence_histogram(tp_kept, fp_kept, tp_disc, fp_disc, threshold, output_path, title="Confidence Distribution", is_inference=False):
    """ Draws a confidence histogram (Kept Global for overall model health overview) """
    plt.figure(figsize=(8, 5))
    bins = np.linspace(0, 1, 21)

    if is_inference:
        # tp_disc contains all below-threshold confidences, tp_kept contains all above-threshold confidences
        if len(tp_disc) > 0:
            plt.hist(tp_disc, bins=bins, alpha=0.4, color='lightskyblue', label='Below Threshold')
        if len(tp_kept) > 0:
            plt.hist(tp_kept, bins=bins, alpha=0.8, color='dodgerblue', label='Valid Detections')
    else:
        if len(tp_disc) > 0:
            plt.hist(tp_disc, bins=bins, alpha=0.4, color='lightgreen', label='Missed TP (Below Thresh)')
        if len(fp_disc) > 0:
            plt.hist(fp_disc, bins=bins, alpha=0.4, color='lightcoral', label='Ignored FP (Below Thresh)')
        if len(tp_kept) > 0:
            plt.hist(tp_kept, bins=bins, alpha=0.8, color='green', label='Valid Hits (TP)')
        if len(fp_kept) > 0:
            plt.hist(fp_kept, bins=bins, alpha=0.8, color='red', label='Critical Errors (FP)')

    plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')

    plt.title(title)
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.xlim([0.0, 1.0])
    plt.legend(loc="upper right")
    plt.savefig(output_path)
    plt.close()

def plot_normalized_heatmap(centers, output_path, title="Normalized Detection Heatmap", cmap='inferno'):
    """ Draws a normalized detection heatmap (Kept Global) """
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

def plot_pr_curve(curves_data, output_path):
    """ Draws a PR curve per class """
    plt.figure(figsize=(8, 5))
    
    # Draw one line per class
    for c_id, data in curves_data.items():
        plt.plot(data['recalls'], data['precisions'], lw=2, label=f"{data['name']} (AP50={data['ap50']:.3f})")
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Multi-Class IoU=0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower left", fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_path)
    plt.close()

def plot_f1_curve(curves_data, output_path):
    """ Draws an F1 curve per class """
    plt.figure(figsize=(8, 5))
    
    # Draw one line per class
    for c_id, data in curves_data.items():
        line, = plt.plot(data['confs'], data['f1s'], lw=2, label=f"{data['name']} (Peak F1={data['best_f1']:.2f} @ {data['best_conf']:.2f})")
        plt.scatter([data['best_conf']], [data['best_f1']], color=line.get_color(), zorder=5)
        
    plt.xlabel('Confidence Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 vs Confidence (Threshold Optimization)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower center", fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_path)
    plt.close()