def process_docstring(app, what, name, obj, options, lines):
    """Process the docstring for classes that inherit from Dataset."""
    try:
        from mirdata.core import Dataset
        if what == 'class' and isinstance(obj, type) and issubclass(obj, Dataset) and obj is not Dataset:
            # Find the module that contains this class
            module_name = obj.__module__
            module = __import__(module_name, fromlist=['INDEXES', 'REMOTES'])
            
            # If the module has INDEXES, add version info
            if hasattr(module, 'INDEXES'):
                indexes = module.INDEXES
                
                # Add version info
                lines.append('')
                lines.append('Available Versions')
                lines.append('-----------------')
                
                for version in indexes.keys():
                    lines.append(f" - **{version}**")
            
            # If the module has REMOTES, add remote files info
            if hasattr(module, 'REMOTES'):
                remotes = module.REMOTES
                
                # Add remote files info
                lines.append('')
                lines.append('Remote Files')
                lines.append('------------')
                
                for remote_name, remote_info in remotes.items():
                    lines.append(f" - **{remote_name}**: {remote_info.url}")
    except (ImportError, AttributeError, TypeError) as e:
        print(f"Warning: Error processing docstring for {name}: {e}")
        pass

def setup(app):
    app.connect('autodoc-process-docstring', process_docstring)
    return {'version': '0.1', 'parallel_read_safe': True}