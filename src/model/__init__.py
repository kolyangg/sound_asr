from src.model.baseline_model import BaselineModel
# from src.model.ds2new2 import DeepSpeech2Model # new v2
from src.model.ds2new2_try import DeepSpeech2Model # new v2
from src.model.ds2_reg import DeepSpeech2ModelReg # new v2
from src.model.conf import ConformerModel # Conformer
# from src.model.ds2_backup import DeepSpeech2Model # old
# from src.model.ds2new import DeepSpeech2Model # new v1



__all__ = [
    "BaselineModel",
    "DeepSpeech2Model",
    "DeepSpeech2ModelReg"
    "ConformerModel"
]
