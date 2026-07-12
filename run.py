"""量化炒股平台 - 桌面应用入口。"""
from __future__ import annotations

import sys
import os

# 确保项目根在 sys.path,便于以 python run.py 直接运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quant.gui.app import main

if __name__ == "__main__":
    main()
