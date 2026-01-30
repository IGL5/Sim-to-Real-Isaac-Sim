# 🚴‍♂️ YOLOv8 Training & Validation Pipeline

Este directorio contiene el conjunto de herramientas completo para procesar los datos sintéticos generados por Isaac Sim, entrenar un detector de objetos **YOLOv8** y auditar los resultados mediante informes visuales.

## 📂 Contenido del Directorio

- `dataset_manager.py`: ETL (Extract, Transform, Load). Convierte etiquetas de formato KITTI a YOLO, gestiona la estructura de carpetas (`train/val/test`) y permite añadir datos incrementalmente.

- `train_YOLO.py`: Script de entrenamiento. Configura automáticamente el entorno para YOLOv8 y exporta el modelo final a ONNX.

- `visualize_results.py`: Herramienta de auditoría y testeo. Genera reportes HTML interactivos, matrices de confusión y visualizaciones de predicciones.

- `report_utils.py`: Librería auxiliar para cálculos matemáticos (IoU, métricas) y generación de gráficos.

## ⚙️ Instalación

Este pipeline requiere librerías específicas de Computer Vision y Data Science. Instálalas con:

```bash
pip install ultralytics opencv-python matplotlib seaborn pandas pyyaml
```

Nota: Se recomienda usar un entorno virtual o Conda para no interferir con el entorno de Isaac Sim si se ejecuta en la misma máquina.

## 🚀 Flujo de Trabajo

### Paso 1: Gestión del Dataset (dataset_manager.py)

Este script busca automáticamente los datos generados en la carpeta ../_output_data/ del repositorio principal.

### Opciones:

1. **Crear Dataset desde cero (Reset):** Borra cualquier dataset anterior y crea una estructura limpia.

```bash
python dataset_manager.py
```

2. **Modo Incremental (Append):** Útil si has generado una nueva tanda de imágenes en Isaac Sim y quieres sumarlas a tu dataset de entrenamiento sin borrar lo que ya tenías. Renombra los archivos con un timestamp para evitar duplicados.

```bash
python dataset_manager.py --append
```

### Paso 2: Entrenamiento (train_YOLO.py)

Descarga el modelo pre-entrenado (YOLOv8 Small por defecto) y realiza el fine-tuning con tus datos.

# Entrenamiento estándar (50 épocas)

```bash
python train_YOLO.py
```

# Personalizar duración
```bash
python train_YOLO.py --epochs 100
```

- **Salida:** Los pesos del modelo se guardan en cyclist_detector/v1_yolov8_small/weights/best.pt.

- **Exportación:** Al finalizar, se genera automáticamente una versión .onnx lista para producción.

### Paso 3: Auditoría y Visualización (visualize_results.py)

Una vez entrenado el modelo, usa esta herramienta para entender qué está pasando.

#### 🕵️ **Modo Auditoría (Dataset de Test)**

Analiza las imágenes del conjunto de test (que tienen etiquetas reales) y compara con la predicción de la IA.

- **Ver solo errores:** Genera carpetas separadas para Falsos Negativos (no vistos) y Falsos Positivos (inventados).

```bash
python visualize_results.py
```

- **Ver todo (Reporte Completo):** Genera imágenes con cajas Verdes (Realidad) y Azules (IA + Confianza). Crea un reporte HTML con mapa de calor y métricas.

```bash
python visualize_results.py --draw_all
```

#### 🌍 **Modo Inferencia (Mundo Real)**

Prueba tu modelo con fotos nuevas que no tienen etiquetas (ej. fotos reales de cámara).

```bash
python visualize_results.py --source /ruta/a/mis/fotos_reales
```

#### 🎥 **Modo Vídeo**

Procesa un vídeo MP4 y genera un vídeo de salida con las detecciones.

```bash
python visualize_results.py --video assets/video_prueba.mp4
```

## 📊 El Reporte HTML (audit_report/)

Si ejecutas el modo auditoría, se generará una carpeta audit_report. Abre el archivo report.html en tu navegador para ver:

Precision/Recall/F1: Métricas de calidad industrial.

Mapa de Calor: ¿Detecta tu modelo solo en el centro de la imagen o cubre bien los bordes?

Histograma de Confianza: ¿Está el modelo demasiado seguro de sus errores?