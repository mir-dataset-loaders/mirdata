def add_versions(module, lines):
    if hasattr(module, 'INDEXES'):
        indexes = module.INDEXES
        lines.append('**Versions:** ' + ' '.join([f'``{v}``' for v in indexes.keys()]))
        lines.append('')


def add_partial_downloads(module, dataset_class, lines):
    remotes = getattr(dataset_class, 'REMOTES', None) or getattr(module, 'REMOTES', {})
    if isinstance(remotes, dict):
        lines.append('')
        lines.append('**Partial Downloads:**')
        lines.append(' '.join([f'``{v}``' for v in remotes.keys()]))
        lines.append('')


def process_docstring(app, what, name, obj, options, lines):
    try:
        from mirdata.core import Dataset

        if what == 'class' and isinstance(obj, type) and issubclass(obj, Dataset) and obj is not Dataset:
            module = __import__(obj.__module__, fromlist=['INDEXES'])
            add_versions(module, lines)

        if what == 'method' and name.endswith('.download'):
            class_path = name.rsplit('.', 1)[0]
            parts = class_path.split('.')
            module_path = '.'.join(parts[:-1])
            class_name = parts[-1]
            module = __import__(module_path, fromlist=[class_name])
            dataset_class = getattr(module, class_name)
            if hasattr(dataset_class, 'REMOTES') or hasattr(module, 'REMOTES'):
                add_partial_downloads(module, dataset_class, lines)

    except (ImportError, AttributeError, TypeError) as e:
        print(f"Warning: Error processing docstring for {name}: {e}")


def setup(app):
    app.connect('autodoc-process-docstring', process_docstring)
    return {'version': '0.1', 'parallel_read_safe': True}
