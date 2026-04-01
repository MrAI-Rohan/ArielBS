def convert_numerics(obj):
        if isinstance(obj, dict):
            return {k: convert_numerics(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numerics(item) for item in obj]
        elif isinstance(obj, str):
            try:
                return float(obj)
            except ValueError:
                return obj
        else:
            return obj