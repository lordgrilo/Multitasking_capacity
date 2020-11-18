import numpy as np



####### analytical functions for Gaussian degree distributions 
def attempt_scipy_func_gauss(x, d,sigma, c):
    return np.abs(x - np.power(1 - ((sigma*sigma*np.log(x)+d)/float(d*x)) * np.exp((d*np.log(x) + 0.5*sigma*sigma*np.power(np.log(x),2))), c-1));


def minimize_gauss_pstar(xs, d, sigma, c):
    vals = list(map(lambda x: attempt_scipy_func_gauss(x, d, sigma, c), xs));
    x0 = (np.min(vals), xs[np.nanargmin(vals)])
    return x0[1]

def rho_gauss(x,d,sigma,c):
    ed01dp0 = (1 - d - sigma*sigma*np.log(x)) * np.exp(d*np.log(x) + 0.5*sigma*sigma*np.log(x)*np.log(x))
    return (d/c)  * (1 - np.power(x,c/float(c-1))) + (ed01dp0);

def gaussian_prediction(ps,sigmas,xs=None,c=2):
#    if xs == None:
#        xs = np.linspace(0,1,1000);
    import time 
    now = time.time()
    rho_stars_gauss = []
    for i,d in enumerate(ps):
        p_stars_gauss = minimize_gauss_pstar(xs, d , sigmas[i], c);
        rho_stars_gauss.append(rho_gauss(p_stars_gauss, d , sigmas[i], c));
    return rho_stars_gauss;

####### analytical functions for gamma degree distributions 
def attempt_scipy_func_gamma(x, gamma,  theta, c):
    return np.abs(x - (1.0  - (1.0/(x*(1.0 * theta*np.log(x)))) * np.power((1.0/(1 - theta*np.log(x))), gamma)))

def minimize_gamma_pstar(xs, gamma, theta, c):
    vals = map(lambda x: attempt_scipy_func_gamma(x, gamma, theta, c), xs);
    x0 = (np.min(vals), xs[np.nanargmin(vals)])
    return x0[1]

def rho_gamma(x,d,c,gamma,theta):
    frac = (1.0/(1 - theta*np.log(x)));
    mod = (1 - frac) * np.power(frac,gamma);
    return (d/c)  * (1 - np.power(x,c/float(c-1))) + mod;

######################################################################


import sympy as sy

def p_star(d,c):
    x = sy.Symbol('x');
    return sy.solve(x - sy.Pow((1-sy.exp(d*(x-1))), c-1), x);

def rho(d,c,p_star):
    return (d/float(c)) * (1-np.power(float(p_star),(c/float(c-1)))) + (1-float(p_star)*d)*np.exp(d*(float(p_star)-1));


###### Generic generating function 

def eval_genfunc(pk,x):
    tot = np.sum(pk.values())
    return np.sum([(pk[k]/float(tot)) * np.power(x,k) for k in pk]);

def eval_genfunc_prime(pk,x):
    tot = np.sum(pk.values())
    return np.sum([k*(pk[k]/float(tot)) * np.power(x,k-1) for k in pk]);    

def attempt_scipy_func_generic(x, d, pk, c):
    return np.abs(x - np.power(1 - (eval_genfunc_prime(pk,np.log(x)) / float(d*x)), c-1));

def minimize_pstar_generic(xs, d, pk, c):
    vals = map(lambda x: attempt_scipy_func_generic(x, d, pk, c), xs);
    x0 = (np.min(vals), xs[np.nanargmin(vals)])
    return x0[1]

def rho_generic(x,d,pk,c):
    M = eval_genfunc(pk,np.log(x));
    Mprime = eval_genfunc_prime(pk,np.log(x));
    mod = M - Mprime;
    return (d/c)  * (1 - np.power(x,c/float(c-1))) + mod;


###### UTILS #####
def plot_results(x,data,norm=None,label=None,marker='s'):
    yy = list(map(lambda x: np.mean(data[x]), sorted(data.keys())));
    std_yy = list(map(lambda x: np.std(data[x]), sorted(data.keys())));
    plt.gca()
    if norm==None:
        plt.errorbar(x, np.array(yy), np.array(std_yy), fmt=marker,ms=10,alpha=0.6, label=label);
    else:
        plt.errorbar(x, np.array(yy)/float(norm), np.array(std_yy)/float(norm), fmt=marker,ms=10,alpha=0.6, label=label);
    return;





