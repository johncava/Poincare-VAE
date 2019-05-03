import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import *
from scipy.stats import truncnorm
from scipy.stats import uniform
from scipy.special import erf
from scipy.misc import comb

epsilon = 1e-8
var = 1
d = 3
c = 1

# Normalization constant for Truncated Normal
Zg = math.sqrt((var*math.pi)/2.0)*(1 + erf((d - 1)*math.sqrt(c)*var/math.sqrt(2)))

# Normalization constant for Hyperbolic radius pdf
Zsum = [((-1)**k)*comb((d-1),k)*math.exp((((d - 1 - 2*k)**2)*c*var)/2.0)*(1 + erf((d-1-2*k)*math.sqrt(c*var)/math.sqrt(2))) for k in list(range(d))]
Zr = math.sqrt((var*math.pi)/2.0)*(1.0/((2*math.sqrt(c))**(d-1)))*sum(Zsum)

# Acceptance-Rejection constant
M = (Zg/Zr)*(1.0/((2*math.sqrt(c))**(d-1)))*math.exp((((d-1)** 2)*c*var)/2.0)

# Sample Truncated Normal
def sample_g():
    rv = truncnorm.rvs(epsilon,math.inf,loc=(d - 1)*math.sqrt(c)*var, scale=var, size=1)
    return rv

# Truncated Normal PDF
def truncnorm_pdf(r):
    return truncnorm.pdf(r, epsilon,math.inf,loc=(d - 1)*math.sqrt(c)*var, scale=var)

# Hyperbolic radius PDF
def hyperbolic_pdf(r):
    if r > 0:
        return (1.0/Zr)*math.exp(-(r**2)/(2*var))*(math.sinh(math.sqrt(c)*r)/math.sqrt(c))**(d-1)
    else:
        return 0

#print(Zg)
#print(Zr)

r = None
accept = False
while not accept:
    r = sample_g()
    u = uniform.rvs(size=1)
    p = hyperbolic_pdf(r)
    g = truncnorm_pdf(r)
    if u < M*(p/g):
        accept = True
    else:
        continue
r = r[0]

# Uniform distribution on the hypersphere
UHS = MultivariateNormal(torch.zeros(d-1), torch.eye(d-1))
alpha = UHS.sample()
alpha = alpha/alpha.norm(2)

mu = torch.randn((1,10))
lambda_mu = (2.0/(1 - c*torch.norm(mu)**2))
z = torch.exp((r/lambda_mu)*alpha)
print(z)





