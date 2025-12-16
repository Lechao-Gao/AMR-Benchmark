"""
AMR-Benchmark 模型定义包
"""
from .CNN1Model import CNN1Model
from .CNN2Model import CNN2Model
from .LSTM2Model import LSTM2Model
from .DAEModel import DAEModel
from .ResNetModel import ResNetModel
from .ICAMCModel import ICAMC as ICAMCModel
from .MCLDNN import MCLDNN
from .PETCGDNN import PETCGDNN

__all__ = ['CNN1Model', 'CNN2Model', 'LSTM2Model', 'DAEModel', 'ResNetModel','GRUModel','ICAMCModel','MCLDNN','PETCGDNN']