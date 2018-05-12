import importlib

def reload_modules(modules):
    for module in modules:
        importlib.reload(module)
