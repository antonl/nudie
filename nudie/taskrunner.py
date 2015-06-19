import sys
import nudie
from PyQt4 import QtGui, QtCore
import multiprocessing
from pathlib import Path

from nudie.pump_probe import main as pp_main
from nudie.dd import main as dd_main
from nudie.tg import main as tg_main
from nudie.stark_tg import main as stark_tg_main
from nudie.phasing import main as phasing_main
from nudie.apply_phase import main as apply_phase_main
from nudie.stark_dd import main as stark_dd_main

class MainWindow(QtGui.QMainWindow):
    def __init__(self, *args, **kwargs):
        QtGui.QMainWindow.__init__(self, *args, **kwargs)
        
        self.setWindowTitle('Nudie Task Runner')
        self.create_main_frame()

        ctx = multiprocessing.get_context('spawn')
        self.workers = ctx.Pool(5)

    def create_main_frame(self):
        self.main_frame = QtGui.QWidget()
        
        s = 'Select a configuration file and then push a button to start an ' +\
            'analysis.'
        inst_lb = QtGui.QLabel(s)

        config_group = QtGui.QGroupBox("Configuration file")
        config_layout = QtGui.QHBoxLayout()
        self.config_lb = QtGui.QLabel()
        config_btn = QtGui.QPushButton("Choose file")
        config_btn.clicked.connect(self.on_choose_file)
        config_layout.addWidget(self.config_lb)
        config_layout.addWidget(config_btn)
        config_group.setLayout(config_layout)

        exp_group = QtGui.QGroupBox("Experiments")
        exp_layout = QtGui.QHBoxLayout()
        self.pp_btn = QtGui.QPushButton("Pump Probe")
        self.pp_btn.clicked.connect(self.on_pump_probe)
        self.tg_btn = QtGui.QPushButton("Transient Grating")
        self.tg_btn.clicked.connect(self.on_tg)
        self.dd_btn = QtGui.QPushButton("2D")
        self.dd_btn.clicked.connect(self.on_dd)
        self.stark_chk = QtGui.QCheckBox("Stark")
        self.stark_chk.setChecked(False)
        self.stark_chk.stateChanged.connect(self.on_stark_check)

        for x in [self.pp_btn, self.tg_btn, self.dd_btn, self.stark_chk]:
            x.setMinimumWidth(120)
            exp_layout.addWidget(x)
            exp_layout.setStretchFactor(x, 1)
        exp_group.setLayout(exp_layout)

        phasing_group = QtGui.QGroupBox("Phasing and such")
        phasing_layout = QtGui.QHBoxLayout()
        self.phasing_btn = QtGui.QPushButton("Phasing")
        self.phasing_btn.clicked.connect(self.on_phasing)
        self.applyphase_btn = QtGui.QPushButton("Apply Phase")
        self.applyphase_btn.clicked.connect(self.on_apply_phase)

        for x in [self.phasing_btn, self.applyphase_btn]:
            phasing_layout.addWidget(x)
        phasing_layout.addStretch(1)
        phasing_group.setLayout(phasing_layout)

        vbox = QtGui.QVBoxLayout()
        for x in [inst_lb, config_group, exp_group, phasing_group]:
            vbox.addWidget(x)
        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)

    def on_stark_check(self, state):
        # No such thing as stark pp
        if state == 2: # checked
            for x in [self.pp_btn]:
                x.setEnabled(False)
            self.dd_btn.setText('Stark 2D')
            self.tg_btn.setText('Stark TG')
        elif state == 0:
            for x in [self.pp_btn]:
                x.setEnabled(True)
            self.dd_btn.setText('2D')
            self.tg_btn.setText('Transient Grating')

    def on_choose_file(self):
        try:
            file = QtGui.QFileDialog.getOpenFileName(self, "Open cfg file", 
                    "", "Nudie config files (*.cfg)") 
            self.config = Path(file)
            self.config_lb.setText(str(self.config.relative_to(
                self.config.cwd())))
        except Exception as e:
            QtGui.QMessageBox.critical(self, "Exception occured", str(e))

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
        app = QtGui.QApplication(sys.argv)
        main = MainWindow()
        main.show()
        app.exec_()
    except Exception as e:
        raise e

if __name__ == '__main__':
    run()
