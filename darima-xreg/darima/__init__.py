import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

package_name = "forecast"

try:
    pkg = importr(package_name)
except:
    robjects.r(f'install.packages("{package_name}")')
    pkg = importr(package_name)

from .model import sarima2ar_model, darima_model
from .dlsa import dlsa_mapreduce
from .forecast import forecast_darima, darima_forec
from .evaluation import eval_func, model_eval
