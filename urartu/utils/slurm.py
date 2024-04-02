def is_submitit_available() -> bool:
    try:
        import submitit  # NOQA

        return True
    except ImportError:
        return False
