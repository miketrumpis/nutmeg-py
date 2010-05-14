from PyQt4 import QtGui
import sys
from nutmeg.vis.tfbeam_manager import *

bfile = '/Users/mike/workywork/nutmeg-py/data/NC_RightDom_Digits_New/matlab_ana/new_fdb_all/s_beamtf1_avg.mat'

def tfbeam_manager_test():
    if QtGui.QApplication.startingUp():
        app = QtGui.QApplication(sys.argv)
    else:
        app = QtGui.QApplication.instance()    

    win = NmTimeFreqWindow((), (), (), [ [-100,100] ]*3)
    bman = win.func_man
    bman.update_beam(bfile)
    sres = tfstats_results.load_tf_snpm_stats(bfile)
    bman.bstats_manager.stats_results = sres
    win.activate()
    win.show()
    return win, app

# hide from Nose
tfbeam_manager_test.__test__ = False
