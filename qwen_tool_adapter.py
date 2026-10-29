"""Thin adapters for Qwen-Agent custom tools."""
import json
from qwen_agent.tools.base import BaseTool

class EvidenceTool(BaseTool):
    def __init__(self, store, name, description, parameters):
        self.store, self.name, self.description, self.parameters = store, name, description, parameters
        super().__init__()

    def call(self, params, **kwargs):
        try:
            args = self._verify_json_format_args(params)
            result = self.store.execute(self.name, args)
        except (ValueError, TypeError) as exc:
            result = {'success': False, 'error': str(exc)}
        return json.dumps(result, ensure_ascii=False)

def object_schema(properties, required):
    return {'type': 'object', 'properties': properties, 'required': required}
