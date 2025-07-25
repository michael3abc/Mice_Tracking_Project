# run.py
import sys
import os
import yaml   
from PyQt5.QtWidgets import QApplication
from gui_display import MainWindow

def main():
    
    cfg_path = r"src\gui\gui_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    app = QApplication(sys.argv)
    window = MainWindow(cfg)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
