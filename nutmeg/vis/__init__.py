__docformat__ = 'restructuredtext'
import os, sys
from PyQt4 import QtGui

## if QtGui.QApplication.startingUp():
##     global_app = QtGui.QApplication(sys.argv)
## else:
##     global_app = QtGui.QApplication.instance()

# register the nutmeg plugin before launching any xip viewers
from xipy.overlay.plugins import register_overlay_plugin, \
     all_registered_plugins

# collect all (existing) plugins to register
from nutmeg.vis.tfbeam_manager import NmTimeFreqWindow, TFBeamManager
register_overlay_plugin(NmTimeFreqWindow)

from nutmeg.vis.tfstats_manager import TimeFreqSnPMaps

import xipy.volume_utils as vu
from xipy.io import load_spatial_image

def plot_tfbeam_3d(beam):
    pass

def plot_tfbeam(beam, stats=None, with3d=False):
    from xipy.vis.ortho_viewer import ortho_viewer
    struct = load_spatial_image(beam.coreg.mrpath)
    sman = TimeFreqSnPMaps(stats_results=stats)
    bbox = vu.world_limits(struct)
    bman = TFBeamManager(bbox, bstats_manager=sman)
    bman.update_beam(beam)


    win = ortho_viewer(image=struct, mayavi_viewer=with3d)
    win.make_tool_from_functional_manager(NmTimeFreqWindow, bman)
    win.show()
##     app.exec_()
##     # XYZ: this is truly ugly
    bman.signal_image_props()
    bman.signal_new_image()
    return win

def with_attribute(a):
    def dec(f):
        def runner(obj, *args, **kwargs):
            if not getattr(obj, a, False):
                return
            return f(obj, *args, **kwargs)
        # copy f's info to runner
        for attr in ['func_doc', 'func_name']:
            setattr(runner, attr, getattr(f, attr))
        return runner
    return dec    
