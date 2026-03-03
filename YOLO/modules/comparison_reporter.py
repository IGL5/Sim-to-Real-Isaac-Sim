import os
import json
from . import plot_generator
from .html_generator import HTMLReportGenerator
import core_visual_utils as cvu

class ComparisonReporter:
    def __init__(self, models_dict):
        self.models_dict = models_dict
        self.project_dir = cvu.PROJECT_DIR
        self.output_dir = os.path.join(os.getcwd(), "comparison_report")
        self.plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        self.data = {}

    def _load_json(self, path):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def load_all_data(self):
        for display_label, info in self.models_dict.items():
            audit_path = info["path"]
            real_model_name = info["model_root_name"] # Real model name
            
            audit_meta = self._load_json(audit_path)
            
            # Search for the rest of metadata in the "vault" using the real model name
            model_root = os.path.join(self.project_dir, real_model_name)
            train_meta_path = os.path.join(model_root, "metadata", "training_metadata.json")
            dataset_meta_path = os.path.join(model_root, "metadata", "dataset_metadata.json")
            
            train_meta = self._load_json(train_meta_path)
            dataset_meta = self._load_json(dataset_meta_path)

            # Save all under the "display_label" for proper HTML display
            self.data[display_label] = {
                "audit": audit_meta,
                "train": train_meta,
                "dataset": dataset_meta
            }

    def generate_plots(self):
        print("📊 Generating comparative plots...")
        models = list(self.data.keys())
        
        # 1. Main Performance Plot (Grouped)
        precision = [self.data[m]["audit"]["metrics"].get("precision", 0) for m in models]
        recall = [self.data[m]["audit"]["metrics"].get("recall", 0) for m in models]
        map50 = [self.data[m]["audit"]["metrics"].get("ap50", 0) for m in models]
        
        plot_generator.plot_comparison_bar(
            models, 
            [precision, recall, map50], 
            ["Precision", "Recall", "mAP@50"], 
            os.path.join(self.plots_dir, "compare_metrics.png"),
            "Detection Performance"
        )

        # 2. AI Confidence Plot (Hits vs Errors)
        conf_tp = [self.data[m]["audit"]["metrics"].get("confidence_stats", {}).get("True_Positives", {}).get("mean", 0) for m in models]
        conf_fp = [self.data[m]["audit"]["metrics"].get("confidence_stats", {}).get("False_Positives", {}).get("mean", 0) for m in models]
        
        plot_generator.plot_comparison_bar(
            models, 
            [conf_tp, conf_fp], 
            ["Confidence Hits (TP)", "Confidence Misses (FP)"], 
            os.path.join(self.plots_dir, "compare_confidence.png"),
            "AI Confidence (Mean)"
        )

        # 3. Generation Cost Plot (Time/Frame)
        gen_times = []
        for m in models:
            ds_meta = self.data[m].get("dataset")
            t = 0.0
            if ds_meta and "sessions" in ds_meta and len(ds_meta["sessions"]) > 0:
                t = ds_meta["sessions"][-1].get("performance", {}).get("average_time_per_frame_seconds", 0.0)
            gen_times.append(t)
            
        plot_generator.plot_simple_bar(
            models, gen_times, "Seconds / Frame", 
            os.path.join(self.plots_dir, "compare_time.png"),
            "Synthetic Data Generation Cost"
        )

    def generate_comparison(self):
        self.load_all_data()
        self.generate_plots()
        
        print("📝 Compiling comparison HTML report...")
        templates_dir = os.path.join(os.getcwd(), "modules", "templates")
        dataset_out_dir = os.path.join(os.getcwd(), "dataset_yolo_output")
        
        # Use our existing generator
        generator = HTMLReportGenerator(templates_dir, self.project_dir, dataset_out_dir)
        
        output_path = os.path.join(self.output_dir, "comparison_report.html")
        generator.generate_comparison_html(output_path, self.data)
        print(f"\n✅ Comparison completed! Report available at: {output_path}")