import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as cm
import numpy as np
from pyart.io.nexrad_archive import read_nexrad_archive
from numpy import genfromtxt
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import convolve1d
import os
from cmap_radar import createMapNorm
import improve_matplot.improve_matplot as im

# function to do convolution (along range dimension)
#----------------------------------
def conv(x, w):
    numk = len(w)
    numx = x.shape[1]
    y = np.ma.masked_all([x.shape[0], numx])

    for i in range(numx-numk):
        numvalid = x[:,i:i+numk].count(axis=1)
        y[numvalid==numk,i+(numk-1)/2] = np.ma.dot(x[numvalid==numk,i:i+numk], w)

    return y

# function to de-alias phidp
#----------------------------------
def dealiasPhiDP(phiDP):
    deal_phi = np.ma.empty([phiDP.shape[0], phiDP.shape[1]])
    deal_phi[phiDP<0.] = 180.+phiDP[phiDP<0.] 
    deal_phi[phiDP>=0.] = phiDP[phiDP>=0.]
    return deal_phi   

# function for smoothing phidp
#----------------------------------
def smPhiDP(phiDP, ran):
    # smooth phiDP field and take derivative
    # calculate lanczos filter weights
    numRan = ran.shape[0]
    numK = 11
    fc = 0.015
    kt = np.linspace(-(numK-1)/2, (numK-1)/2, numK)
    w = np.sinc(2.*kt*fc)*(2.*fc)*np.sinc(kt/(numK/2))

    #smoothPhiDP = convolve1d(phiDP, w, axis=1, mode='constant', cval=-999.)
    smoothPhiDP = conv(phiDP, w)
    #smoothPhiDP = np.ma.masked_where(smoothPhiDP==-999., smoothPhiDP)

    return smoothPhiDP

# function for estimating kdp
#----------------------------------
def calculateKDP(phiDP, ran):
    # smooth phiDP field and take derivative
    numRan = ran.shape[0]
    kdp = np.ma.masked_all(phiDP.shape)
    smoothPhiDP = smPhiDP(phiDP, ran)

    # take derivative of kdp field
    winLen = 7
    rprof = ran[0:winLen*2-1]/1000.

    for i in range(numRan-winLen*3):
        numvalid = smoothPhiDP[:,i:i+winLen*2-1].count(axis=1)
        max_numv = np.max(numvalid)
        if max_numv==(winLen*2-1):
            kdp[numvalid==(winLen*2-1),i+winLen] = 0.5*np.polyfit(rprof,
                smoothPhiDP[numvalid==(winLen*2-1),i:i+winLen*2-1].transpose(), 1)[0]
    return kdp

# function for creating color map
#----------------------------------
def createCmap(mapname):
    fil = open(mapname+'.rgb')
    cdata = genfromtxt(fil,skip_header=2)
    cdata = cdata/256
    cmap = cm.ListedColormap(cdata, mapname)
    fil.close()
    return cmap

# function to convert x,y to lon,lat
#-----------------------------------
def xy2latlon(x, y, lat0, lon0):
    km2deg = 110.62
    lat = y/km2deg+lat0
    lon = x/(km2deg*np.cos(np.pi*lat0/180.))+lon0
    return lat, lon

# define dictionary for color table names, ranges
#-----------------------------------------------------------------------
color_info = {'zh':{'cmap_file':'zh2_map', 'vmin':-10., 'vmax':21.},
              'zdr':{'cmap_file':'zdr_map', 'vmin':-3.2, 'vmax':9.2},
              'phidp':{'cmap_file':'zh2_map', 'vmin':0., 'vmax':128.},
              'rhohv':{'cmap_file':'phv_map', 'vmin':0.45, 'vmax':1.07},
              'vel':{'cmap_file':'vel2_map', 'vmin':-31., 'vmax':31.},
              'kdp':{'cmap_file':'kdp_map', 'vmin':-1.6, 'vmax':4.6},
              'sw':{'cmap_file':'zh2_map', 'vmin':0., 'vmax':6.2}}

# open file
#-----------------------------------
print 'Opening file...'

yyyy = '2018'
mm = '08'
dd = '18'

hh = '21'
mn = '26'
ss = '35'
site = 'KILX'

raddir = '/graupel/s0/rss5116/radar_data/nexrad/kilx/'
filename = raddir+'{}{}{}{}_{}{}{}_V06'.format(site, yyyy, mm, dd,
                                               hh, mn, ss)
print filename
rad = read_nexrad_archive(filename)
rad_sw = rad.extract_sweeps([6])

# get variables
#-----------------------------------
print rad_sw.fields
elev_p = rad_sw.elevation['data']
azi_p = 90.-rad_sw.azimuth['data']
ran = rad_sw.range['data']
ref_p = rad_sw.fields['reflectivity']['data']
zdr_p = rad_sw.fields['differential_reflectivity']['data']
phidp_p = rad_sw.fields['differential_phase']['data']
vel_p = rad_sw.fields['velocity']['data']
rhohv_p = rad_sw.fields['cross_correlation_ratio']['data']
radlat = rad_sw.latitude['data'][0]
radlon = rad_sw.longitude['data'][0]

dims = ref_p.shape
numradials = dims[0]+1
numgates = dims[1]

# expand radially to remove no data spike
elev = np.ma.empty([numradials])
azi = np.ma.empty([numradials])
ref = np.ma.empty([numradials, numgates])
zdr = np.ma.empty([numradials, numgates])
phidp = np.ma.empty([numradials, numgates])
vel = np.ma.empty([numradials, numgates])
rhohv = np.ma.empty([numradials, numgates])

elev[0:numradials-1] = elev_p
elev[numradials-1] = elev_p[0]
azi[0:numradials-1] = azi_p
azi[numradials-1] = azi_p[0]
ref[0:numradials-1,:] = ref_p
ref[numradials-1,:] = ref_p[0]
zdr[0:numradials-1,:] = zdr_p
zdr[numradials-1,:] = zdr_p[0]
phidp[0:numradials-1,:] = phidp_p
phidp[numradials-1,:] = phidp_p[0]
vel[0:numradials-1,:] = vel_p
vel[numradials-1,:] = vel_p[0]
rhohv[0:numradials-1,:] = rhohv_p
rhohv[numradials-1,:] = rhohv_p[0]

angle = np.mean(elev)

# mask data by rhohv and threshold
#-----------------------------------------------
ref = np.ma.masked_where(rhohv<0.4, ref)
zdr = np.ma.masked_where(rhohv<0.4, zdr)
phidp = np.ma.masked_where(rhohv<0.4, phidp)
vel = np.ma.masked_where(rhohv<0.4, vel)
rhohv = np.ma.masked_where(rhohv<0.4, rhohv)

zdr = np.ma.masked_where(ref<-5., zdr)
phidp = np.ma.masked_where(ref<-5., phidp)
vel = np.ma.masked_where(ref<-5., vel)
rhohv = np.ma.masked_where(ref<-5., rhohv)
ref = np.ma.masked_where(ref<-5., ref)

# calculate kdp
#-----------------------------------------------
print 'Calculating KDP...'
phidp = dealiasPhiDP(phidp)
kdp = calculateKDP(phidp, ran)
kdp = np.ma.masked_where(ref<-5., kdp)

# calculate x and y coordinates (wrt beampath) for plotting
#-----------------------------------------------------------
ran_2d = np.tile(ran,(numradials,1))
azi.shape = (azi.shape[0], 1)
azi_2d = np.tile(azi,(1,numgates))

radz = 10.
erad = np.pi*angle/180.

ke = 4./3.
a = 6378137.

# beam height and beam distance
zcor = np.sqrt(ran_2d**2.+(ke*a)**2.+2.*ran_2d*ke*a*np.sin(erad))-ke*a+radz
scor = ke*a*np.arcsin(ran_2d*np.cos(erad)/(ke*a+zcor))/1000.

xcor = scor*np.cos(np.pi*azi_2d/180.)
ycor = scor*np.sin(np.pi*azi_2d/180.)

# create lists of fields and their attributes in order
fields = [ref, zdr, phidp, rhohv]
names = ['zh', 'zdr', 'phidp', 'rhohv']
titles = ['Z$\sf{_H}$', 'Z$\sf{_{DR}}$', '$\\Psi\sf{_{DP}}$', '$\sf{\\rho_{HV}}$']
units = ['dBZ', 'dB', 'degrees', '']

# create basic PPI plot
#----------------------------
# common plot elements
fig = plt.figure()
im.adjustFonts(mpl)
cblb_fsize = 24
cbti_fsize = 20
axtl_fsize = 30
axlb_fsize = 24
axti_fsize = 20

ds = 80.
xcen = 0.
ycen = 0.
xmin = xcen-ds
xmax = xcen+ds
ymin = ycen-ds
ymax = ycen+ds

xlabel = 'X-distance (km)'
ylabel = 'Y-distance (km)'

# smooth zdr for contour plotting
sig = 2.5
sm_zdr = gaussian_filter(zdr, sig)
zdr_lev = 1.2

# get range ring of data
r0 = 50.
ri = np.argmin(np.abs(r0*1.e3-ran))
z0 = np.sqrt(ran[ri]**2.+(ke*a)**2.+2.*ran[ri]*ke*a*np.sin(erad))-ke*a+radz
s0 = ke*a*np.arcsin(ran[ri]*np.cos(erad)/(ke*a+z0))/1000.
print z0/1.e3, s0
phi_ring = np.linspace(0., 2.*np.pi, 201)
xring = s0*np.cos(phi_ring)
yring = s0*np.sin(phi_ring)

zh_ring = ref[:,ri]
zdr_ring = zdr[:,ri]
phidp_ring = phidp[:,ri]
vel_ring = vel[:,ri]
rhohv_ring = rhohv[:,ri]

# write out azimuthal data
fname = '{}{}{}{}_{}{}_azi.txt'.format(site, yyyy, mm, dd, hh, mn)
f = open(fname, "w")
f.write('# height (km)\tdistance (km)\n')
f.write('{:.1f}\t{:.1f}\n'.format(z0, s0))
f.write('# {}\t{}\t{}\t{}\t{}\t{}\n'.format('azi', 'zh', 'zdr', 'phidp', 'rhohv', 'vel'))

for i in range(numradials):
    f.write('{:.1f}\t{:.1f}\t{:.2f}\t{:.1f}\t{:.3f}\t{:.1f}\n'.format(azi[i,0], zh_ring[i], zdr_ring[i],
                                                              phidp_ring[i], rhohv_ring[i], vel_ring[i]))
f.close()

# loop through panels
for i in range(4):
    ax = fig.add_subplot(2,2,i+1)
    cmapn, normn = createMapNorm(names[i], cinfo=color_info)
    plt.plot(xring, yring, 'k--', lw=3.)
    plt.pcolormesh(xcor, ycor, fields[i], cmap=cmapn, norm=normn)
    cb = plt.colorbar()
    cb.set_label(units[i], fontsize=cblb_fsize)
    cb_la = [ti.get_text().replace('$', '') for ti in cb.ax.get_yticklabels()]
    cb.ax.set_yticklabels(cb_la, fontsize=cbti_fsize)

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_title(titles[i], x=0.0, y=1.02,
                 horizontalalignment='left', fontsize=axtl_fsize)
    ax.set_xlabel(xlabel, fontsize=axlb_fsize)
    ax.set_ylabel(ylabel, fontsize=axlb_fsize)
    # change tick mark sizes and fonts
    ax.set_xticklabels(ax.get_xticks(), fontsize=axti_fsize)
    ax.set_yticklabels(ax.get_yticks(), fontsize=axti_fsize)
    ax.grid(color='k', linestyle=(0.5, [2,6]), linewidth=1.)

# save image as .png
#-------------------------------
title = '{} - {}/{}/{} - {}:{} UTC - {:.1f} deg. PPI'.format(site, yyyy, mm, dd,
                                                             hh, mn, float(angle))
plt.suptitle(title, fontsize=36)
#plt.subplots_adjust(top=0.89, hspace=0.15, wspace=0.)
imgname = yyyy+mm+dd+'_'+hh+mn+'_'+site.lower()+'.png'
plt.savefig(imgname, format='png', dpi=250)

# crop out white space from figure
os.system('convert -trim '+imgname+' '+imgname)
