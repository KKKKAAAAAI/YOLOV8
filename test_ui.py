import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QAction, QMenuBar, QMessageBox

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 设置窗口标题
        self.setWindowTitle("Menu Action Example")

        # 创建菜单栏
        menubar = self.menuBar()

        # 创建 FITNESS 菜单
        fitness_menu = menubar.addMenu("FITNESS")
        fitness_actions = [
            "Deadlift", "Lat Pulldown", "Squats", "Bench Press",
            "Dumbbel Shoulder Press", "Dumbbell Lateral Raises"
        ]
        for action_name in fitness_actions:
            action = QAction(action_name, self)
            action.triggered.connect(self.handle_action_triggered)
            fitness_menu.addAction(action)

        # 创建 REHABILITATION 菜单
        rehab_menu = menubar.addMenu("REHABILITATION")
        rehab_actions = [
            "Shoulder rehabilitation", "Knee rehabilitation", "Lumbar spine rehabilitation"
        ]
        for action_name in rehab_actions:
            action = QAction(action_name, self)
            action.triggered.connect(self.handle_action_triggered)
            rehab_menu.addAction(action)

    def handle_action_triggered(self):
        # 获取发送信号的 action
        action = self.sender()
        action_text = action.text()

        # 显示 action 的文本
        QMessageBox.information(self, "Action Triggered", f"Action triggered: {action_text}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
