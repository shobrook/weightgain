try:
    from weightgain.Adapter import Adapter
    from weightgain.Dataset import Dataset
    from weightgain.Model import Model
except ImportError:
    from .Adapter import Adapter
    from .Dataset import Dataset
    from .Model import Model
