import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from UI.MainWindows import MainWinddow
import sys

app = QApplication(sys.argv)
win = MainWinddow()
win.show()
def close_fun():
    sys.exit(0)
    
win.hook_close_win(close_fun)
sys.exit(app.exec_()) 

