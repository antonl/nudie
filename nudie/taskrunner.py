import sys
from qtpy import QtGui, QtCore, QtWidgets
import nudie
import multiprocessing
from pathlib import Path

from nudie.pump_probe import main as pp_main
from nudie.dd import main as dd_main
from nudie.tg import main as tg_main
from nudie.stark_tg import main as stark_tg_main
from nudie.phasing import main as phasing_main
from nudie.apply_phase import main as apply_phase_main
from nudie.stark_dd import main as stark_dd_main
from nudie.linear_stark import main as linear_stark_main

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        
        self.setWindowTitle('Nudie Task Runner')
        self.create_main_frame()

        ctx = multiprocessing.get_context('spawn')
        self.workers = ctx.Pool(5)

    def create_main_frame(self):
        self.main_frame = QtWidgets.QWidget()
        
        s = 'Select a configuration file and then push a button to start an ' +\
            'analysis.'
        inst_lb = QtWidgets.QLabel(s)

        config_group = QtWidgets.QGroupBox("Configuration file")
        config_layout = QtWidgets.QHBoxLayout()
        self.config_lb = QtWidgets.QLabel()
        config_btn = QtWidgets.QPushButton("Choose file")
        config_btn.clicked.connect(self.on_choose_file)
        config_layout.addWidget(self.config_lb)
        config_layout.addWidget(config_btn)
        config_group.setLayout(config_layout)

        exp_group = QtWidgets.QGroupBox("Experiments")
        exp_layout = QtWidgets.QHBoxLayout()
        self.pp_btn = QtWidgets.QPushButton("Pump Probe")
        self.pp_btn.clicked.connect(self.on_pump_probe)
        self.tg_btn = QtWidgets.QPushButton("Transient Grating")
        self.tg_btn.clicked.connect(self.on_tg)
        self.dd_btn = QtWidgets.QPushButton("2D")
        self.dd_btn.clicked.connect(self.on_dd)
        self.linear_btn = QtWidgets.QPushButton("Linear Absorption")
        self.linear_btn.clicked.connect(self.on_linear)
        self.linear_btn.setEnabled(False)
        self.stark_chk = QtWidgets.QCheckBox("Stark")
        self.stark_chk.setChecked(False)
        self.stark_chk.stateChanged.connect(self.on_stark_check)

        for x in [self.pp_btn, self.tg_btn, self.dd_btn, self.linear_btn, self.stark_chk]:
            x.setMinimumWidth(120)
            exp_layout.addWidget(x)
            exp_layout.setStretchFactor(x, 1)
        exp_group.setLayout(exp_layout)

        phasing_group = QtWidgets.QGroupBox("Phasing and such")
        phasing_layout = QtWidgets.QHBoxLayout()
        self.phasing_btn = QtWidgets.QPushButton("Phasing")
        self.phasing_btn.clicked.connect(self.on_phasing)
        self.applyphase_btn = QtWidgets.QPushButton("Apply Phase")
        self.applyphase_btn.clicked.connect(self.on_apply_phase)

        for x in [self.phasing_btn, self.applyphase_btn]:
            phasing_layout.addWidget(x)
        phasing_layout.addStretch(1)
        phasing_group.setLayout(phasing_layout)

        vbox = QtWidgets.QVBoxLayout()
        for x in [inst_lb, config_group, exp_group, phasing_group]:
            vbox.addWidget(x)
        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)

    def on_stark_check(self, state):
        # No such thing as stark pp
        # Currently no normal linear absorption code for this ui
        if state == 2: # checked
            for x in [self.pp_btn]:
                x.setEnabled(False)
            for x in [self.linear_btn]:
                x.setEnabled(True)
            self.dd_btn.setText('Stark 2D')
            self.tg_btn.setText('Stark TG')
            self.linear_btn.setText('Linear Stark')
        elif state == 0:
            for x in [self.pp_btn]:
                x.setEnabled(True)
            for x in [self.linear_btn]:
                x.setEnabled(False)
            self.dd_btn.setText('2D')
            self.tg_btn.setText('Transient Grating')
            self.linear_btn.setText('Linear Absorption')

    def on_choose_file(self):
        try:
            file, ftype = QtWidgets.QFileDialog.getOpenFileName(self, "Open cfg file", "", "Nudie config files (*.cfg)")
            self.config = Path(file)
            self.config_lb.setText(str(self.config.relative_to(
                self.config.cwd())))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Exception occured", str(e))

    def on_pump_probe(self):
        self.pp_btn.setEnabled(False)
        self.pp_btn.setText("Running pp")
        self.workers.apply_async(pp_main, [self.config],{}, self.pump_probe_done,
            self.pump_probe_done)

    def pump_probe_done(self, result):
        self.pp_btn.setEnabled(True)
        self.pp_btn.setText("Pump Probe")

    def on_tg(self):
        self.tg_btn.setEnabled(False)
        if self.stark_chk.isChecked():
            self.tg_btn.setText("Running stark tg")
            self.workers.apply_async(stark_tg_main, [self.config],{}, self.tg_done,
                self.tg_done)
        else:
            self.tg_btn.setText("Running TG")
            self.workers.apply_async(tg_main, [self.config],{}, self.tg_done,
                self.tg_done)

    def tg_done(self, result):
        if self.stark_chk.isChecked():
            self.tg_btn.setText("Stark TG")
        else:
            self.tg_btn.setText("Transient Grating")
        self.tg_btn.setEnabled(True)

    def on_dd(self):
        self.dd_btn.setEnabled(False)
        if self.stark_chk.isChecked():
            self.dd_btn.setText("Running stark dd")
            self.workers.apply_async(stark_dd_main, [self.config],{}, self.dd_done, 
                    self.dd_done)
        else:
            self.dd_btn.setText("Running 2D")
            self.workers.apply_async(dd_main, [self.config],{}, self.dd_done, 
                    self.dd_done)

    def dd_done(self, result):
        self.dd_btn.setEnabled(True)
        if self.stark_chk.isChecked():
            self.dd_btn.setText("Stark 2D")
        else:
            self.dd_btn.setText("2D")
            
    def on_linear(self):
        self.linear_btn.setEnabled(False)
        self.linear_btn.setText("Running Linear Stark")
        self.workers.apply_async(linear_stark_main, [self.config],{},self.linear_done,
            self.linear_done)

            
    def linear_done(self, result):
        if self.stark_chk.isChecked():
            self.linear_btn.setEnabled(True)
            self.linear_btn.setText("Linear Stark")
        else:
            self.linear_btn.setEnabled(False)
            self.linear_btn.setText("Linear Absorption")
            
    def on_phasing(self):
        self.phasing_btn.setEnabled(False)
        self.phasing_btn.setText("Running phasing")
        self.workers.apply_async(phasing_main, [self.config],{}, self.phasing_done,
            self.phasing_done)

    def phasing_done(self, result):
        self.phasing_btn.setEnabled(True)
        self.phasing_btn.setText("Phasing")

    def on_apply_phase(self):
        self.applyphase_btn.setEnabled(False)
        self.applyphase_btn.setText("Running apply phase")
        self.workers.apply_async(apply_phase_main, [self.config],{},
                self.apply_phase_done, self.apply_phase_done)
    def apply_phase_done(self, result):
        self.applyphase_btn.setEnabled(True)
        self.applyphase_btn.setText("Apply phase")

def run():
    try:
        app = QtWidgets.QApplication(sys.argv)
        main = MainWindow()
        main.show()
        app.exec_()
    except Exception as e:
        raise e

if __name__ == '__main__':
    run()
