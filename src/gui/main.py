# run.py
import sys, torch
import os
import yaml   
from PyQt5.QtWidgets import QApplication
from gui_display import MainWindow
from openvino.runtime import Core

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
os.environ["QT_SCALE_FACTOR"] = "1.0"

def main():
    print(torch.xpu.is_available())
    ie = Core()
    print(ie.available_devices)  # 應該要看到類似 ['CPU', 'GPU', …]
    
    cfg_path = r"src\gui\gui_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    app = QApplication(sys.argv)
    window = MainWindow(cfg)    
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
