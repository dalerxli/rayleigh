import numpy as np
from scipy.special import beta
import matplotlib.pyplot as plt

# integral function
def i_n(n, a, b):
    ival = beta(a+n,b)/beta(a,b)
    return ival

# calculate angular moments
a = 2.
b = 18.

i1 = i_n(1,a,b)
i2 = i_n(2,a,b)
i3 = i_n(3,a,b)
i4 = i_n(4,a,b)

a1 = 4*(i2-i1)+1.
a2 = 2*(i1-i2)
a3 = 8*(2*i4-4*i3+3*i2-i1)+1.
a4 = 6*(i4-2*i3+i2)
a5 = 2*(-4*i4+8*i3-5*i2+i1)

print a1, a2, a3, a4, a5
print np.arccos(1.-2.*0.3)*180./np.pi, np.arccos(1.-2.*0.7)*180./np.pi

# plot beta distribution
x = np.linspace(0., 1., 201)
f = x**(a-1.)*(1.-x)**(b-1.)/beta(a,b)
plt.plot(x, f, 'r--', lw=3.)
plt.savefig('beta.png')
