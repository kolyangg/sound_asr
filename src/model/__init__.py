from src.model.baseline_model import BaselineModel
# from src.model.ds2new2 import DeepSpeech2Model # new v2
from src.model.ds2new2_try import DeepSpeech2Model # new v2
# from src.model.ds2_reg import DeepSpeech2ModelReg # new v2
from src.model.conf import ConformerModel # Conformer
# from src.model.ds2_backup import DeepSpeech2Model # old
# from src.model.ds2new import DeepSpeech2Model # new v1
from src.model.ds2_newreg import DeepSpeech2ModelReg # new v2
from src.model.ds2_newreg2 import DeepSpeech2ModelReg2 # new v2
from src.model.ds2_30Dec import DeepSpeech2ModelSimple # new v2
from src.model.ds2_30Dec2 import DeepSpeech2ModelSimple2
from src.model.ds2new_31Dec import DeepSpeech2ModelNew



__all__ = [
    "BaselineModel",
    "DeepSpeech2Model",
    "DeepSpeech2ModelReg",
    "DeepSpeech2ModelReg2",
    "DeepSpeech2ModelSimple",
    "DeepSpeech2ModelSimple2",
    "DeepSpeech2ModelNew",
    "ConformerModel"
]
