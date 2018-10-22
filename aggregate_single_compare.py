import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import orientation as ort
import single_particle as sp

# maxwell-garnett mixing
def maxwell_garnett(eps_inc, vfrac):
    eps_fac = (eps_inc-1.)/(eps_inc+2.)
    eps_mix = (1.+2.*vfrac*eps_fac)/(1.-vfrac*eps_fac)
    return eps_mix

# set particle and radar properties
wavl = 32.1
k = 2.*np.pi/wavl
eps_ice = complex(3.16835, 0.0089)
dmax = 2.
thick = 1.99

# generate aggregate
asp_inc = 1./8.
nfrac = 201
vfrac_inc = 10.**(np.linspace(-2., 0., nfrac))
amp = 11
mod = 0.5
a_inc = (amp-2)*mod+1
b_inc = amp*(1-mod)+2*mod-1

angmom_inc = ort.angmom_beta_inc(a_inc, b_inc)
eps_a, eps_c = sp.eps_spheroid_beta(eps_ice, vfrac_inc, angmom_inc, asp_inc, 1.)

# calculate radar variables (unit volume particle)
alpha_a, alpha_a, alpha_c = sp.ani_oblate_polz(eps_a, eps_c, dmax/2., thick/2.)
vol_obl = np.pi/6.*dmax**2.*thick*vfrac_inc
alpha_a = alpha_a/vol_obl
alpha_c = alpha_c/vol_obl

sigh_agg = k**2.*np.abs(alpha_a)**2.
sigv_agg = k**2.*np.abs(alpha_c)**2.
zdr_agg = 10.*np.log10(sigh_agg/sigv_agg)
kdp_agg = 180.e-3/wavl*np.real(alpha_a-alpha_c)
sigh_agg = np.log10(sigh_agg)
sigv_agg = np.log10(sigv_agg)
#print sigh_agg, zdr_agg, kdp_agg

# compare to isolated pristine crystals
#aobl_a, aobl_a, aobl_c = sp.oblate_polz(eps_ice, asp_inc/2., 1./2.)
aprl_c, aprl_a, aprl_a = sp.prolate_polz(eps_ice, 1./2., asp_inc/2.)
vol_sph = np.pi/6.*asp_inc**2.
#aobl_a = aobl_a/vol_sph
#aobl_c = aobl_c/vol_sph
aprl_a = aprl_a/vol_sph
aprl_c = aprl_c/vol_sph

angmom_pr = ort.angmom_beta(a_inc, b_inc)
#acov_arr = ort.avg_polcov(angmom_pr, aobl_a, aobl_c)
acov_arr = ort.avg_polcov(angmom_pr, aprl_a, aprl_c)
zdr, ldr, kdp, rhohv, zhh, rhoxh = ort.radar(wavl, acov_arr)
sigh = np.log10(k**2.*acov_arr[0])
sigv = np.log10(k**2.*acov_arr[1])

# plot radar variables
fig = plt.figure(1)
ax = fig.add_subplot(2,2,1)
plt.plot(vfrac_inc, zdr*vfrac_inc/vfrac_inc, 'r-', lw=3.)
plt.plot(vfrac_inc, zdr_agg, 'b-', lw=3.)
#ax.set_ylim([0., 1.])
#ax.set_xlim([0.05, 1.])

ax = fig.add_subplot(2,2,2)
plt.plot(vfrac_inc, np.log10(kdp)*vfrac_inc/vfrac_inc, 'r-', lw=3.)
plt.plot(vfrac_inc, np.log10(kdp_agg), 'b-', lw=3.)
#ax.set_ylim([0., 1.])
#ax.set_xlim([0.05, 1.])

ax = fig.add_subplot(2,2,3)
plt.plot(vfrac_inc, sigh*vfrac_inc/vfrac_inc, 'r-', lw=3.)
plt.plot(vfrac_inc, sigh_agg, 'b-', lw=3.)
ax.set_ylim([-2., -0.5])
#ax.set_xlim([0.05, 1.])

ax = fig.add_subplot(2,2,4)
plt.plot(vfrac_inc, sigv*vfrac_inc/vfrac_inc, 'r-', lw=3.)
plt.plot(vfrac_inc, sigv_agg, 'b-', lw=3.)
ax.set_ylim([-2., -0.5])
#ax.set_xlim([0.05, 1.])

plt.savefig('agg_sing_compare.png')
