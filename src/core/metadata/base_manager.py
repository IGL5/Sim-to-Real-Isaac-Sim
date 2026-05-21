import json
from datetime import datetime
from pathlib import Path

class BaseMetadataManager:
    """
    Base class (Data Access Object / Repository) for all metadata managers.
    It exclusively handles reading, writing, and merging JSON dictionaries.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = {}
        self._load_existing()

    def _load_existing(self):
        """ Loads the JSON if it already exists to avoid overwriting data from other passes. """
        self.data = self.read_json(self.filepath)

    @staticmethod
    def read_json(filepath):
        """Reads an external JSON file and returns its dictionary. Returns {} if not found or error."""
        path = Path(filepath)
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ [BaseMetadataManager] Error reading JSON from {filepath}: {e}")
        return {}

    def update_section(self, section_name, payload):
        """
        Updates or creates a main section in the JSON (e.g. 'performance', 'hardware').
        """
        if section_name not in self.data:
            self.data[section_name] = {}
            
        self.data[section_name].update(payload)

    def set_timestamp(self, key_name="last_updated"):
        """Records the exact time of the last modification."""
        self.data[key_name] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def commit(self):
        """
        Persists the data in memory to the physical JSON file.
        Creates the necessary folders if they do not exist.
        """
        path = Path(self.filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=4)
            print(f"💾 Metadata successfully saved to: {self.filepath}")
        except Exception as e:
            print(f"❌ [BaseMetadataManager] Critical error saving {self.filepath}: {e}")