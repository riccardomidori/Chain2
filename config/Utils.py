import pprint

def auto_str(cls):
    def __str__(self):
        params = {
            k: v for k, v in vars(self).items()
        }
        return "%s\n%s" % (
            type(self).__name__,
            pprint.pformat(params, indent=2)
        )

    cls.__str__ = __str__
    return cls
