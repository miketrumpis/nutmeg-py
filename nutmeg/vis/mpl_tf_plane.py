import matplotlib as mpl
import matplotlib.cm
from matplotlib.lines import Line2D
from matplotlib.image import AxesImage
import numpy as np

now = True

def try_or_pass(default=None):
    def dec(func):
        def to_run(*args, **kwargs):
            try:
                a = func(*args, **kwargs)
                if a is not None:
                    return a
            except:
                return default
        return to_run
    return dec

class TimeFreqPlaneFigure(object):

    def __init__(self, fig, t_ax, f_ax):
        self.fig = fig
        self.canvas = fig.canvas
        if not fig.axes:
            self.ax = self.fig.add_subplot(111)
            self.ax.set_aspect('auto', adjustable='box')
            self.ax.set_position([.1, .1, .85, .85])
        else:
            self.ax = fig.axes[0]
        self._cx_id = self.canvas.mpl_connect('button_press_event',
                                              self.tf_selection_callback)
        self.selected_tf_plane = None
        self.cbar = None
        self.reset_plane(t_ax, f_ax)        

    def reset_plane(self, t_ax, f_ax):
        self.tf_plane = None
        self.t_ax = t_ax
        self.f_ax = f_ax
        self.selected_tf_index = (0,0)
        self.xlim = (t_ax[0], t_ax[-1])
        self.ylim = (f_ax[0], f_ax[-1])
        self.clear_plane()
    
    @try_or_pass()
    def _get_xlim(self):
        return self.ax.get_xlim()
    @try_or_pass()
    def _set_xlim(self, xlim):
        self.ax.set_xlim(xlim)
        self.draw()
    @try_or_pass()
    def _get_ylim(self):
        return self.ax.get_ylim()
    @try_or_pass()
    def _set_ylim(self, ylim):
        self.ax.set_ylim(ylim)
        self.draw()
    @try_or_pass()
    def _set_cmap(self, cmap):
        self.tf_plane.set_cmap(cmap)
        self.draw()
    @try_or_pass()
    def _get_cmap(self):
        return self.tf_plane.get_cmap()        
    @try_or_pass()
    def _set_norm(self, norm):
        self.tf_plane.set_norm(norm)
        self.draw()
    @try_or_pass()
    def _get_norm(self):
        return self.tf_plane.norm
##     @try_or_pass()
##     def _set_alpha(self, alpha):
##         self.tf_plane.set_alpha(alpha)
##         self.draw()
##     @try_or_pass(default=1)
##     def _get_alpha(self):
##         return self.tf_plane.get_alpha()
    xlim = property(_get_xlim, _set_xlim)
    ylim = property(_get_ylim, _set_ylim)
    cmap = property(_get_cmap, _set_cmap)
    norm = property(_get_norm, _set_norm)
##     alpha = property(_get_alpha, _set_alpha)

    def _t_index(self, t):
        less_than = self.t_ax[self.t_ax <= t]
        return np.argmax(less_than)

    def _f_index(self, f):
        less_than = self.f_ax[self.f_ax <= f]
        return np.argmax(less_than)
    
    def get_plane_index(self, t, f):
        return (self._t_index(t), self._f_index(f))

    def update_tf_plane(self, plane, **kwargs):
        """Update the entire pcolor plot with the new tf plane "plane". Also
        potentially updates the tf plane properties.
        """
        cmap = kwargs.get('cmap', self.cmap)
        norm = kwargs.get('norm', self.norm)
        ax = self.ax
        if plane is not None:
            # update the mapped scalars
            if not self.tf_plane:
                self.tf_plane = ax.pcolor(self.t_ax, self.f_ax, plane,
                                          cmap=cmap, norm=norm,
                                          edgecolors='k')
                # everything's set, so skip the rest of the function
                self.update_selected_tf(self.selected_tf_index)
            else:
                self.tf_plane.set_array(plane.flatten())
        # update the mapping properties, if they're in kwargs
        if self.tf_plane and ('cmap' in kwargs or 'norm' in kwargs):
            self.tf_plane.set_norm(norm)
            self.tf_plane.set_cmap(cmap)
            if not self.cbar:                
                try:
                    cax = self.fig.axes[1]
                    cax.hold(True)
                except IndexError:
                    cax, kw = mpl.colorbar.make_axes(self.ax,
                                                     orientation='vertical')
                    self.fig.sca(self.ax)
                self.cbar = mpl.colorbar.Colorbar(cax, self.tf_plane,
                                                  orientation='vertical')
                # not sure what this does for anybody
                self.tf_plane.set_colorbar(self.cbar, cax)

            else:
                self.cbar.set_clim(self.tf_plane.get_clim())
                self.cbar.set_cmap(cmap)
##                 if mpl.__version__
                self.cbar.update_bruteforce(self.tf_plane)
        # finally, if the tf_plane is not in ax.collections, put it there
        if self.tf_plane and self.tf_plane not in ax.collections:
            ax.add_collection(self.tf_plane)
        # only update tf point is there is a new scalar to plot
        if plane is not None:
            self.update_selected_tf(self.selected_tf_index)
        else:
            self.draw()

    def update_selected_tf(self, index):
        """Highlight the rectangle at the tf point "index", and update state
        """
        ax = self.ax
        p_obj = self.tf_plane
        if p_obj is None:
            return
        try:
            a = p_obj.get_array().reshape(
                len(self.f_ax)-1, len(self.t_ax)-1 #, 1, 1
                )
        except ValueError:
            # array is apparently stale, so don't update anything
            return
        value = np.array([[ a[index] ]])

        if self.selected_tf_plane and self.selected_tf_plane in ax.collections:
            ax.collections.remove(self.selected_tf_plane)
        f_idx, t_idx = index
        mini_t_ax = self.t_ax[t_idx:t_idx+2]
        mini_f_ax = self.f_ax[f_idx:f_idx+2]
        self.selected_tf_plane = ax.pcolor(
            mini_t_ax, mini_f_ax, value,
            edgecolors='r', linewidth=2, cmap=self.cmap, alpha=1,
            norm=self.tf_plane.norm
            )
                
        self.selected_tf_index = index
        
        self.xlim = (self.t_ax[0], self.t_ax[-1])
        self.ylim = (self.f_ax[0], self.f_ax[-1])
        self.draw() #(when=now)

    def tf_selection_callback(self, event):
        if not event.inaxes:
            return
        t_index, f_index = self.get_plane_index(event.xdata, event.ydata)
        self.update_selected_tf((f_index, t_index))

    def clear_plane(self):
        if not self.tf_plane:
            return
        shape = ( len(self.f_ax)-1, len(self.t_ax)-1 )
        self.update_tf_plane(np.ma.masked_all(shape),
                             norm=mpl.colors.Normalize(0,1))

    def resize(self, x, y):
        """ x,y are pixel quantities
        """
        dpi = self.fig.get_dpi()
        xi, yi = float(x)/dpi, float(y)/dpi
        self.fig.set_figsize_inches(xi, yi)
        self.draw()
    
    def draw(self, when=not now):
        if when==now:
            self.canvas.draw()
        else:
            self.canvas.draw_idle()
    
if __name__=='__main__':
    import sys
    import matplotlib as mpl
    mpl.use('Qt4Agg')
    import matplotlib.cm
    import matplotlib.figure
    import matplotlib.pyplot as pp
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as Canvas
    import cProfile, pstats
    fig = pp.figure()
    t_ax = np.linspace(-100,100,51)
    f_ax = np.array([0, 10, 40, 80, 160.])
    tf_fig = TimeFreqPlaneFigure(fig, t_ax, f_ax)
    def update_data_cb(ev):
        if ev.inaxes:
            return
        a = np.random.randn(4,50)
        norm = mpl.colors.normalize(a.min(), a.max())
        tf_fig.update_tf_plane(a, norm=norm)

    tf_fig.update_tf_plane(np.arange(4*50).reshape(4,50),
                           cmap=mpl.cm.hot,
                           norm=mpl.colors.normalize(0,199))
    tf_fig.canvas.mpl_connect('button_press_event',
                              update_data_cb)
    pp.show()


##     a = np.random.randn(4,50)
##     norm = mpl.colors.normalize(a.min(), a.max())
##     cProfile.runctx('tf_fig.update_tf_plane(a, norm=norm)', globals(), locals(), 'pcolor.prof')

## ## ##     import matplotlib.pyplot as pp
## ## ##     cProfile.runctx('pp.show()', globals(), locals(), 'pcolor.prof')
##     s = pstats.Stats('pcolor.prof')
##     s.strip_dirs().sort_stats('cumulative').print_stats()
