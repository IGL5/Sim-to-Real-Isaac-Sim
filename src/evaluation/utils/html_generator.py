from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from src.core import config

class HTMLReportGenerator:
    def __init__(self):
        """El generador es ahora una Vista Pura. Solo sabe renderizar diccionarios."""
        self.env = Environment(loader=FileSystemLoader(config.TEMPLATES_DIR))

    def generate_audit_html(self, file_output_path, context):
        template = self.env.get_template('audit_template.html')
        html_content = template.render(context)
        with open(file_output_path, "w", encoding='utf-8') as f:
            f.write(html_content)
        print(f"✅ Audit HTML Report generated at: {file_output_path}")

    def generate_inference_html(self, file_output_path, context):
        template = self.env.get_template('inference_template.html')
        html_content = template.render(context)
        with open(file_output_path, "w", encoding='utf-8') as f:
            f.write(html_content)
        print(f"✅ Inference HTML Report generated at: {file_output_path}")

    def generate_comparison_html(self, file_output_path, context):
        template = self.env.get_template('compare_template.html')
        html_content = template.render(context)
        with open(file_output_path, "w", encoding='utf-8') as f:
            f.write(html_content)
        print(f"✅ Comparison HTML Report generated at: {file_output_path}")