import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def plot_confusion_matrix(confusion_pairs, class_names, output_path):
    """ Draws a dynamic Multi-Class confusion matrix including Background """
    if not confusion_pairs:
        return

    # Obtener todas las clases únicas involucradas, incluyendo -1 (Fondo)
    unique_classes = set([c for pair in confusion_pairs for c in pair])
    
    # Ordenar clases (0, 1, 2...) y dejar el -1 al final
    sorted_classes = sorted([c for c in unique_classes if c != -1])
    if -1 in unique_classes:
        sorted_classes.append(-1)
        
    labels = [class_names.get(c, f"Clase {c}") if c != -1 else "Fondo (BG)" for c in sorted_classes]
    
    # Crear matriz vacía de NxN
    size = len(sorted_classes)
    matrix = np.zeros((size, size), dtype=int)
    
    # Mapear el ID de clase a la fila/columna de la matriz
    idx_map = {c: i for i, c in enumerate(sorted_classes)}
    
    # Rellenar la matriz contando los pares (Real, Predicho)
    for real_c, pred_c in confusion_pairs:
        matrix[idx_map[real_c], idx_map[pred_c]] += 1
        
    # Dibujar
    plt.figure(figsize=(max(6, size*1.2), max(5, size*1.2)))
    ax = sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                     xticklabels=labels, yticklabels=labels)
    plt.title('Matriz de Confusión Multi-Clase')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción de YOLO')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90, va="center")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_confidence_histogram(tp_kept, fp_kept, tp_disc, fp_disc, threshold, output_path, title="Confidence Distribution"):
    """ Draws a confidence histogram (Kept Global for overall model health overview) """
    plt.figure(figsize=(8, 5))
    bins = np.linspace(0, 1, 21)

    plt.hist(tp_disc, bins=bins, alpha=0.4, color='lightgreen', label='Missed TP (Below Thresh)')
    plt.hist(fp_disc, bins=bins, alpha=0.4, color='lightcoral', label='Ignored FP (Below Thresh)')
    plt.hist(tp_kept, bins=bins, alpha=0.8, color='green', label='Valid Hits (TP)')
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
    
    # Dibujar una línea por cada clase
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
    
    # Dibujar una línea por cada clase
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