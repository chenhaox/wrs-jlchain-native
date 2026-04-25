__all__ = ["JLChain", "NativeKDLBackend", "create_backend"]


def __getattr__(name):
    if name == "JLChain":
        from .jlchain import JLChain

        return JLChain
    if name in ("NativeKDLBackend", "create_backend"):
        from .backends import NativeKDLBackend, create_backend

        return {"NativeKDLBackend": NativeKDLBackend, "create_backend": create_backend}[name]
    raise AttributeError(f"module 'wrs_jlchain_native' has no attribute {name!r}")
