import yt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

def plt_multipanel(ds, kind='slice', axis='z',
                   fields=['nH', 'xn', 'G0prime0', 'G0prime1']):
    """
    Make 2x2 multi-panel plot using yt
    Based on
    http://yt-project.org/doc/cookbook/complex_plots.html#multipanel-with-axes-labels
    
    Parameters
    ----------
       ds : yt Dataset
       kind : string
           slice or projection
       axis : string
           x or y or z
       fields: list_like
           List of fields
    
    Returns
    -------
       p : yt.visualization.plot_window.AxisAlignedSlicePlot or
           yt.visualization.plot_window.ProjectionPlot
           yt plot object
    """
    
    if kind == 'slice':
        p = yt.SlicePlot(ds, axis, fields)
    elif kind == 'projection':
        p = yt.ProjectionPlot(ds, axis, fields, weight_field='cell_volume')
        
    fig = plt.figure()
    grid = AxesGrid(fig, (0.075,0.075,0.85,0.85),
                    nrows_ncols=(2,2), axes_pad=(1.2,0.1),
                    label_mode="1", share_all=True,
                    cbar_location="right", cbar_mode="each",
                    cbar_size="4%", cbar_pad="2%")

    for i, field in enumerate(fields):
        plot = p.plots[field]
        plot.figure = fig
        plot.axes = grid[i].axes
        plot.cax = grid.cbar_axes[i]

    p._setup_plots()
    
    # Set clim of G0prime_EUV equal to that of G0prime_FUV
    if ('gas','G0prime0') in p.plots.keys() and \
       ('gas','G0prime1') in p.plots.keys():
        clim = p.plots[('gas','G0prime1')].image.get_clim()
        p.plots[('gas','G0prime0')].image.set_clim(clim)
    
    if ('athena','xn') in p.plots.keys():
        p.plots[('athena','xn')].image.set_clim(1e-8,1.0)

    for i, field in enumerate(fields):
        grid[i].axes.get_xaxis().get_major_formatter().set_scientific(False)
        grid[i].axes.get_yaxis().get_major_formatter().set_scientific(False)

    return p
