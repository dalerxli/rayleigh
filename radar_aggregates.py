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
eps_ice = complex(3.16835, 0.0089)
dmax = 2.
thick = 1.5

# calculate polarizabilities for aggregates with inclusions
asp_inc = 8.
nfrac = 201
vfrac_inc = 10.**(np.linspace(np.log10(0.05), 0., nfrac))
a_inc = 1.
b_inc = 31.
angmom_inc = ort.angmom_beta_inc(a_inc, b_inc)
eps_a, eps_c = sp.eps_spheroid_beta(eps_ice, vfrac_inc, angmom_inc, asp_inc, 1.)
alpha_a, alpha_a, alpha_c = sp.ani_oblate_polz(eps_a, eps_c, dmax/2., thick/2.)

# compare to those with spherical inclusions
eps_mix = maxwell_garnett(eps_ice, vfrac_inc)
alpha_a_sp, alpha_a_sp, alpha_c_sp = sp.oblate_polz(eps_mix, dmax/2., thick/2.)

# set orientation distribution
amp = 18.
mod = 0.
a = (amp-2)*mod+1
b = amp*(1-mod)+2*mod-1

angmom = ort.angmom_beta(a, b)

# calculate radar variables
alp_cov_arr = ort.avg_polcov(angmom, alpha_a, alpha_c)
alp_cov_arr_sp = ort.avg_polcov(angmom, alpha_a_sp, alpha_c_sp)
zdr, ldr, kdp, rhohv, zh, rhoxh = ort.radar(wavl, alp_cov_arr)
zdr_sp, ldr_sp, kdp_sp, rhohv_sp, zh_sp, rhoxh = ort.radar(wavl, alp_cov_arr_sp)
#print zdr, ldr, kdp, rhohv, zh

# plot radar variables
fig = plt.figure(1)
ax = fig.add_subplot(2,2,1)
plt.plot(vfrac_inc, zdr, 'r-', lw=3.)
plt.plot(vfrac_inc, zdr_sp, 'b-', lw=3.)
#ax.set_ylim([0., 1.])
ax.set_xlim([0.05, 1.])

ax = fig.add_subplot(2,2,2)
plt.plot(vfrac_inc, np.log10(kdp), 'r-', lw=3.)
plt.plot(vfrac_inc, np.log10(kdp_sp), 'b-', lw=3.)
#ax.set_ylim([0., 1.])
ax.set_xlim([0.05, 1.])

ax = fig.add_subplot(2,2,3)
plt.plot(vfrac_inc, ldr, 'r-', lw=3.)
plt.plot(vfrac_inc, ldr_sp, 'b-', lw=3.)
#ax.set_ylim([0., 1.])
ax.set_xlim([0.05, 1.])

ax = fig.add_subplot(2,2,4)
plt.plot(vfrac_inc, rhohv, 'r-', lw=3.)
plt.plot(vfrac_inc, rhohv_sp, 'b-', lw=3.)
ax.set_ylim([0.8, 1.05])
ax.set_xlim([0.05, 1.])

plt.savefig('radar_aggregates.png')
