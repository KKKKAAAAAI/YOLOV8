# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI\form\mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(503, 362)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton_start = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_start.sizePolicy().hasHeightForWidth())
        self.pushButton_start.setSizePolicy(sizePolicy)
        self.pushButton_start.setMinimumSize(QtCore.QSize(128, 48))
        self.pushButton_start.setMaximumSize(QtCore.QSize(48, 48))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(12)
        self.pushButton_start.setFont(font)
        self.pushButton_start.setObjectName("pushButton_start")
        self.verticalLayout.addWidget(self.pushButton_start)
        self.pushButton_restart = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_restart.sizePolicy().hasHeightForWidth())
        self.pushButton_restart.setSizePolicy(sizePolicy)
        self.pushButton_restart.setMinimumSize(QtCore.QSize(128, 48))
        self.pushButton_restart.setMaximumSize(QtCore.QSize(48, 48))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(12)
        self.pushButton_restart.setFont(font)
        self.pushButton_restart.setObjectName("pushButton_restart")
        self.verticalLayout.addWidget(self.pushButton_restart)
        self.pushButton_scale = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_scale.sizePolicy().hasHeightForWidth())
        self.pushButton_scale.setSizePolicy(sizePolicy)
        self.pushButton_scale.setMinimumSize(QtCore.QSize(128, 48))
        self.pushButton_scale.setMaximumSize(QtCore.QSize(48, 48))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(12)
        self.pushButton_scale.setFont(font)
        self.pushButton_scale.setObjectName("pushButton_scale")
        self.verticalLayout.addWidget(self.pushButton_scale)
        self.pushButton_end = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_end.sizePolicy().hasHeightForWidth())
        self.pushButton_end.setSizePolicy(sizePolicy)
        self.pushButton_end.setMinimumSize(QtCore.QSize(128, 48))
        self.pushButton_end.setMaximumSize(QtCore.QSize(48, 48))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(12)
        self.pushButton_end.setFont(font)
        self.pushButton_end.setObjectName("pushButton_end")
        self.verticalLayout.addWidget(self.pushButton_end)
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textBrowser.sizePolicy().hasHeightForWidth())
        self.textBrowser.setSizePolicy(sizePolicy)
        self.textBrowser.setMinimumSize(QtCore.QSize(128, 0))
        self.textBrowser.setMaximumSize(QtCore.QSize(128, 16777215))
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayout.addWidget(self.textBrowser)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox.sizePolicy().hasHeightForWidth())
        self.comboBox.setSizePolicy(sizePolicy)
        self.comboBox.setMinimumSize(QtCore.QSize(128, 48))
        self.comboBox.setMaximumSize(QtCore.QSize(128, 48))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(12)
        self.comboBox.setFont(font)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.horizontalLayout.addWidget(self.comboBox)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.label_select = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_select.sizePolicy().hasHeightForWidth())
        self.label_select.setSizePolicy(sizePolicy)
        self.label_select.setMinimumSize(QtCore.QSize(100, 48))
        self.label_select.setMaximumSize(QtCore.QSize(300, 48))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(12)
        self.label_select.setFont(font)
        self.label_select.setAlignment(QtCore.Qt.AlignCenter)
        self.label_select.setObjectName("label_select")
        self.horizontalLayout.addWidget(self.label_select)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.label_image = QtWidgets.QLabel(self.centralwidget)
        self.label_image.setMinimumSize(QtCore.QSize(320, 240))
        self.label_image.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.label_image.setText("")
        self.label_image.setObjectName("label_image")
        self.verticalLayout_2.addWidget(self.label_image)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 503, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.actionDeadlift = QtWidgets.QAction(MainWindow)
        self.actionDeadlift.setObjectName("actionDeadlift")
        self.actionLat_Pulldown = QtWidgets.QAction(MainWindow)
        self.actionLat_Pulldown.setObjectName("actionLat_Pulldown")
        self.actionSquats = QtWidgets.QAction(MainWindow)
        self.actionSquats.setObjectName("actionSquats")
        self.actionBench_Press = QtWidgets.QAction(MainWindow)
        self.actionBench_Press.setObjectName("actionBench_Press")
        self.actionDumbbel_Shoulder_Press = QtWidgets.QAction(MainWindow)
        self.actionDumbbel_Shoulder_Press.setObjectName("actionDumbbel_Shoulder_Press")
        self.actionDumbbell_Lateral_Raises = QtWidgets.QAction(MainWindow)
        self.actionDumbbell_Lateral_Raises.setObjectName("actionDumbbell_Lateral_Raises")
        self.actionShoulder_rehabilitation = QtWidgets.QAction(MainWindow)
        self.actionShoulder_rehabilitation.setObjectName("actionShoulder_rehabilitation")
        self.actionKnee_rehabilitation = QtWidgets.QAction(MainWindow)
        self.actionKnee_rehabilitation.setObjectName("actionKnee_rehabilitation")
        self.actionLumbar_spine_rehabilitation = QtWidgets.QAction(MainWindow)
        self.actionLumbar_spine_rehabilitation.setObjectName("actionLumbar_spine_rehabilitation")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_start.setText(_translate("MainWindow", "START"))
        self.pushButton_restart.setText(_translate("MainWindow", "RESTART"))
        self.pushButton_scale.setText(_translate("MainWindow", "SCALE"))
        self.pushButton_end.setText(_translate("MainWindow", "END"))
        self.comboBox.setItemText(0, _translate("MainWindow", "camera 1"))
        self.comboBox.setItemText(1, _translate("MainWindow", "camera 2"))
        self.label_select.setText(_translate("MainWindow", "REHABILITATION"))
        self.actionDeadlift.setText(_translate("MainWindow", "Deadlift"))
        self.actionLat_Pulldown.setText(_translate("MainWindow", "Lat Pulldown"))
        self.actionSquats.setText(_translate("MainWindow", "Squats"))
        self.actionBench_Press.setText(_translate("MainWindow", "Bench Press"))
        self.actionDumbbel_Shoulder_Press.setText(_translate("MainWindow", "Dumbbel Shoulder Press"))
        self.actionDumbbell_Lateral_Raises.setText(_translate("MainWindow", "Dumbbell Lateral Raises"))
        self.actionShoulder_rehabilitation.setText(_translate("MainWindow", "Shoulder rehabilitation"))
        self.actionKnee_rehabilitation.setText(_translate("MainWindow", "Knee rehabilitation"))
        self.actionLumbar_spine_rehabilitation.setText(_translate("MainWindow", "Lumbar spine rehabilitation"))

