
class KPointContainer:
    """Simple container for storing k-dependent quantities."""

    def __init__(self, **kwargs):
        """Init attributes."""
 
        self.set(**kwargs)

    def set(self, **kwargs):
        """Set attributes from the provided keyword arguments."""

        for key, value in kwargs.items():
            
            assert isinstance(key, str)
            setattr(self, key, value)
