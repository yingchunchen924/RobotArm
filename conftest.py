"""pytest 配置：把 src/ 加入 import 路径，使 `import robotarm...` 可用。"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
