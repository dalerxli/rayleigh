import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import orientation as ort
import single_particle as sp

# cosine transform
def fc(v):
    ang = np.arccos(1.-2.*v)
    return ang

# open obs
data = np.genfromtxt('KILX20180818_2126_azi.txt', skip_header=3)
azi_ob = data[:,0]
zh_ob = data[:,1]
zdr_ob = data[:,2]
phidp_ob = data[:,3]
rhohv_ob = data[:,4]
vel_ob = data[:,5]

# make azimuth positive and sort obs
az_offs = -85.
azi_ob = azi_ob+az_offs
azi_ob[azi_ob<0.] = azi_ob[azi_ob<0.]+360.
azsv = np.argsort(azi_ob)
azi_ob = azi_ob[azsv]
zh_ob = zh_ob[azsv]
zdr_ob = zdr_ob[azsv]
phidp_ob = phidp_ob[azsv]
rhohv_ob = rhohv_ob[azsv]
vel_ob = vel_ob[azsv]

# mask data
zh_ob = np.ma.masked_where(rhohv_ob<0.7, zh_ob)
zdr_ob = np.ma.masked_where(rhohv_ob<0.7, zdr_ob)
phidp_ob = np.ma.masked_where(rhohv_ob<0.7, phidp_ob)
rhohv_ob = np.ma.masked_where(rhohv_ob<0.7, rhohv_ob)
vel_ob = np.ma.masked_where(rhohv_ob<0.7, vel_ob)

# set particle and radar properties
wavl = 100.
eps_ice = complex(7., 1.)
dmax = 0.2
thick = 0.05

# calculate polarizabilities
#alpha_a, alpha_a, alpha_c = sp.oblate_polz(eps_ice, dmax/2., thick/2.)
alpha_c, alpha_a, alpha_a = sp.prolate_polz(eps_ice, dmax/2., thick/2.)
print alpha_c, alpha_a

# set orientation distribution
amp = 11
ang_mode = np.pi*0.6
mod = (1.-np.cos(ang_mode))/2.
a = (amp-2)*mod+1
b = amp*(1-mod)+2*mod-1

phi = azi_ob*np.pi/180.
#angmom = ort.angmom_fixed(ang_mode, phi)
angmom = ort.angmom_beta_phi(a, b, phi)

# calculate radar variables
alpha_cov = ort.avg_polcov(angmom, alpha_a, alpha_c)
avar_hh = alpha_cov[0]
avar_vv = alpha_cov[1]
avar_hv = alpha_cov[2]
acov_hh_vv = alpha_cov[3]
acov_hh_hv = alpha_cov[4]
acov_vv_hv = alpha_cov[5]
adp = alpha_cov[6]

# simultaneous tr
psi_dp = -70.*np.pi/180.
svar_h, svar_v, scov_hv = ort.cov_sim(alpha_cov, psi_dp)
zdr, ldr, kdp, rhohv, zh, rhoxh = ort.radar(wavl, svar_h, svar_v,
                                                avar_hv, scov_hv, acov_hh_hv,
                                                acov_vv_hv, adp)
phi_dp = 180./np.pi*np.angle(scov_hv)
zh = zh-1.

# plot distribution
plt.figure(0)
v = np.linspace(0., 1., 1000)
theta = np.arccos(1.-2.*v)
beta = ort.beta_dist(v, a, b)

plt.plot(theta*180./np.pi, beta, 'b-', lw=1.)
ax = plt.gca()
#ax.set_ylim([0., 1.])
ax.set_xlim([0., 180.])
plt.savefig('dist.png')

# plot radar variables
fig = plt.figure(1)
ax = fig.add_subplot(2,2,1)
plt.plot(phi*180./np.pi, zdr, 'r-', lw=3.)
plt.plot(azi_ob, zdr_ob, 'b-', lw=3.)
#ax.set_ylim([0., 1.])
ax.set_xlim([0., 360.])

ax = fig.add_subplot(2,2,2)
plt.plot(phi*180./np.pi, phi_dp, 'r-', lw=3.)
plt.plot(azi_ob, phidp_ob, 'b-', lw=3.)
#ax.set_ylim([-180., 0.])
ax.set_xlim([0., 360.])

ax = fig.add_subplot(2,2,3)
plt.plot(phi*180./np.pi, rhohv, 'r-', lw=3.)
plt.plot(azi_ob, rhohv_ob, 'b-', lw=3.)
ax.set_ylim([0., 1.1])
ax.set_xlim([0., 360.])

ax = fig.add_subplot(2,2,4)
plt.plot(phi*180./np.pi, zh, 'r-', lw=3.)
plt.plot(azi_ob, zh_ob, 'b-', lw=3.)
#ax.set_ylim([0., 1.1])
ax.set_xlim([0., 360.])

plt.savefig('compare_obs.png')

# plot velocity
plt.figure(2)
plt.plot(data[:,0], data[:,5], 'r--', lw=3.)
plt.savefig('velocity.png')

