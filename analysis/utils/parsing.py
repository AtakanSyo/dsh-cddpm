import re
from typing import Tuple, Optional

def parse_parameters(filename: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse names like: SAMPLE_dust_x_500_0_nH_1_00e_10.png
    -> x = 500.0, nH = 1.00e10
    """
    try:
        m = re.search(r'x_([0-9]+(?:_[0-9]+)?)_nH_([0-9]+(?:_[0-9]+)?)e([_+-]?\d+)\.npy$', filename)
        if not m:
            return None, None
        x_part, base_part, exp_part = m.groups()
        x_val = float(x_part.replace('_', '.'))
        base_str = base_part.replace('_', '.')
        exp_str  = exp_part.replace('_', '')
        nH_val = float(f"{base_str}e{exp_str}")
        return x_val, nH_val
    except Exception:
        return None, None