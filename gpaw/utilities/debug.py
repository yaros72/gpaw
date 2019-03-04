def frozen(cls):
    """Make sure attributes do not appear out of nowhere.

    From:

        https://stackoverflow.com/questions/3603502/
        prevent-creating-new-attributes-outside-init
    """
    def frozensetattr(self, key, value):
        import inspect
        if not hasattr(self, key) and inspect.stack()[1][3] != '__init__':
            raise AttributeError('Class {} is frozen. Cannot set {}'
                                 .format(cls.__name__, key))
        else:
            self.__dict__[key] = value

    cls.__setattr__ = frozensetattr
    return cls
