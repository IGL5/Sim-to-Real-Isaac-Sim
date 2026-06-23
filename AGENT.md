# Guía de Desarrollo para Agentes AI (.claudemd)

¡Hola! Si estás leyendo esto, eres un agente de IA asignado para trabajar en este repositorio. Este documento te guiará rápidamente por el stack tecnológico, la arquitectura actual, las normas de estilo y los antipatrones que debes respetar estrictamente.

---

## 🛠️ Stack Tecnológico
* **Lenguaje**: Python 3.10+ (entorno Conda `sim2real_env`).
* **Deep Learning**: PyTorch, Ultralytics YOLOv8.
* **Simulación**: NVIDIA Isaac Sim (generación de datasets sintéticos).
* **Visión por Computador & Math**: OpenCV (`cv2`), NumPy, Matplotlib.
* **Motor de Renderizado**: Jinja2 (para compilación estática de reportes).
* **Frontend**: HTML5 semántico, CSS vainilla (guardado en `/templates`), Chart.js para visualización de métricas.

---

## 📂 Arquitectura del Proyecto

El código está estructurado bajo principios de Clean Architecture y separación estricta de responsabilidades:

```mermaid
graph TD
    subgraph Core ["Capa Core (src/core/)"]
        Config["config.py (Constantes y Rutas)"]
        Builders["metadata/ (Builders / DAOs)"]
    end
    subgraph UI ["Capa Presentación"]
        Templates["templates/ (HTML/CSS Jinja)"]
        HtmlGen["html_generator.py (Renderizador puro)"]
    end
    subgraph Runners ["Capa Aplicación (src/evaluation/ & src/training/)"]
        TrainYOLO["train_YOLO.py"]
        VisualizeResults["visualize_results.py (CLI Entrypoint)"]
        CompareModels["compare_models.py"]
        Reporters["utils/*_reporter.py (Cálculo y DTOs)"]
    end

    VisualizeResults --> Reporters
    CompareModels --> Reporters
    Reporters --> Builders
    Reporters --> HtmlGen
    HtmlGen --> Templates
    Builders --> Config
```

### Componentes Clave:
1. **`src/core/config.py`**: Centraliza todas las rutas físicas del proyecto (definidas como objetos `Path`) y parámetros como thresholds (`CONF_THRESHOLD`, `IOU_THRESHOLD`).
2. **`src/core/metadata/` (Patrón Repository/DAO)**:
   * `BaseMetadataManager`: Gestiona lectura/escritura física de JSONs.
   * `AuditMetadata`, `TrainMetadata`, `DatasetMetadata`, `InferenceMetadata`: Subclases (Builders) que aíslan el esquema interno JSON y exponen el método `.get_html_summary()`.
   * **DTOs**: `.get_html_summary()` traduce las estructuras jerárquicas complejas en un diccionario aplanado y tipado óptimo para las plantillas HTML.
3. **`src/evaluation/utils/`**:
   * `audit_reporter.py` e `inference_reporter.py`: Realizan la lógica de procesamiento de inferencia y guardan metadatos.
   * `comparison_reporter.py`: Consolida la comparación de múltiples modelos.
   * `html_generator.py`: Instancia el motor Jinja2 puro, consumiendo DTOs para renderizar las vistas en `/templates`.

---

## 📏 Normas de Estilo y Buenas Prácticas

* **Separación de Responsabilidades**: Las plantillas HTML/CSS en `/templates` deben ser puras. No calcules promedios, ni manipules cadenas en el HTML. Pásale a Jinja2 un DTO limpio y ya calculado.
* **Paths Centralizados**: Jamás hardcodees rutas a mano en la lógica (ej. `"data/01_raw"`). Importa siempre `src.core.config as config` y concatena usando operadores de `pathlib.Path`.
* **Seguridad en la Vista**: Asegura fallbacks con filtros de Jinja2 o comprobaciones condicionales (ej. `{{ data[m].audit.global.optimal_conf if data[m].audit.global.optimal_conf is defined else '0.000' }}`). Las páginas deben renderizarse incluso si algún archivo de metadatos está ausente o incompleto.
* **Comentarios y Docstrings**: Conserva los comentarios y docstrings originales del autor. El código se documenta en inglés para el CLI e interno, pero los reportes visuales finales se muestran en un formato amigable.

---

## 🚫 Antipatrones a Evitar (Red Flags)

1. **Importación Directa de `json`**: **No importes el módulo `json`** en scripts principales o reporteros. El único archivo autorizado para importar e interactuar directamente con `json` es `src/core/metadata/base_manager.py`. Todo lo demás debe usar Builders (`AuditMetadata(path)`, etc.).
2. **Cálculos en Plantillas**: No realices agrupaciones de datos ni lógicas de negocio en los archivos de plantilla de Jinja2. Pre-calcula la información en Python en la capa del reportero.
3. **Ignorar YOLO Params**: Al llamar a `model.predict()`, especifica siempre `project` y `name` apuntando a `config.EVALUATION_OUTPUT_DIR` para evitar que Ultralytics ensucie el directorio raíz con carpetas `runs/detect/*`.
4. **Mutación de Metadatos sin Commit**: No modifiques los diccionarios internos de metadatos directamente. Llama a los métodos setter provistos por las clases Builder de metadatos y asegúrate de ejecutar `.commit()` para persistir los cambios a disco.

---

## 🐙 Flujo de Git y Commits

* **Mensajes Semánticos**: Usa prefijos claros al realizar commits:
  * `feat:` para nuevas funcionalidades (ej. guardar el tamaño del dataset).
  * `fix:` para correcciones de bugs (ej. corregir el tipo de parámetro en renderizado).
  * `refactor:` para cambios estructurales de código sin alterar comportamiento.
  * `docs:` para actualizaciones de documentación (como este archivo).
* **Commits Atómicos**: No mezcles cambios visuales/HTML en el mismo commit que refactorizaciones estructurales de Python. Realiza commits pequeños y progresivos para facilitar revertir cambios si es necesario.
