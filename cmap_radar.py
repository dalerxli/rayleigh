import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as cm

# define dictionary for color table names, ranges
color_info = {'zh':{'cmap_file':'zh2_map', 'vmin':-30., 'vmax':32.},
              'zdr':{'cmap_file':'zdr_map', 'vmin':-2.4, 'vmax':6.9},
              'phidp':{'cmap_file':'zh2_map', 'vmin':0., 'vmax':186.},
              'rhohv':{'cmap_file':'phv_map', 'vmin':0.74, 'vmax':1.05},
              'vel':{'cmap_file':'vel2_map', 'vmin':-31., 'vmax':31.},
              'kdp':{'cmap_file':'kdp_map', 'vmin':-1.6, 'vmax':4.6},
              'sw':{'cmap_file':'zh2_map', 'vmin':0., 'vmax':6.2}}

print color_info['zh']['cmap_file']

# create colormap from .rgb file
def createCmap(mapname):
    fil = open(mapname+'.rgb')
    cdata = np.genfromtxt(fil,skip_header=2)
    cdata = cdata/256
    cmap = cm.ListedColormap(cdata, mapname)
    fil.close()
    return cmap

# get map and norm for field
def createMapNorm(field_name, **kwargs):
    # check for color info kwargs
    if 'cinfo' in kwargs:
        color_info = kwargs['cinfo']

    # create color map from file
    path = '/home/meteo/rss5116/research/radar_code/'
    cmap_file = color_info[field_name]['cmap_file']
    field_map = createCmap(path+cmap_file)

    # define number of color levels
    numlevs = 32
    numcolors = numlevs-1
    cinds = np.linspace(0., 254., numcolors).astype(int)

    vmin = color_info[field_name]['vmin']
    vmax = color_info[field_name]['vmax']

    field_cols = field_map.colors[cinds]
    field_levs = np.linspace(vmin, vmax, numlevs)

    field_mapn, field_norm = cm.from_levels_and_colors(field_levs, field_cols, extend='neither')
    return field_mapn, field_norm
