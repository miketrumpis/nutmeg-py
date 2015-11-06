__docformat__ = 'restructuredtext'

from traits.etsconfig.etsconfig import ETSConfig
ETSConfig.toolkit = 'qt4'


def register_xipy_tools():
    # register the nutmeg plugin before launching the ortho viewer
    from xipy.overlay.plugins import register_overlay_plugin, \
         all_registered_plugins

    # collect all (existing) plugins to register
    from nutmeg.vis.tfbeam_manager import NmTimeFreqWindow, TFBeamManager
    register_overlay_plugin(NmTimeFreqWindow)
