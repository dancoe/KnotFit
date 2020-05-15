#check what you actually use, delete unused imports
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.modeling.functional_models import Sersic2D
from reproject import reproject_interp
from astropy.cosmology import FlatLambdaCDM
from pylab import contour, savefig
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.visualization import LogStretch, ManualInterval, ImageNormalize, PowerStretch
import math
import emcee
from scipy.optimize import minimize
from mpl_toolkits.axes_grid1 import make_axes_locatable


def main():
    indir = 'whl0137'
    imfile = 'hlsp_relics_hst_wfc3ir-60mas_whl0137-08_f110w_drz.fits'
    imfile = os.path.join(indir,imfile)
    modeldir = 'whl0137/testruns/z6twopt'
    dflx = os.path.join(modeldir, 'dx_z6.fits')
    dfly = os.path.join(modeldir, 'dy_z6.fits')

    ax, ay, imdataHST = initDeflection_Image(imfile, dflx, dfly, 
                                             zlens=0.566, zsource_in=6.0, zsource_out=6.2)
    
    magnifinv = makeMagnificationMap(ax, ay)
    
    # Define coordinate grid
    yy, xx = np.indices(magnifinv.shape)
    # Source position grid
    xss = xx - ax
    yss = yy - ay
    
    xlo, xhi = 2700, 3100
    ylo, yhi = 1800, 2100
    
    imstamp = imdataHST[ylo:yhi, xlo:xhi]
    
    star = starGen(imdataHST, starLoc=(2232, 1622))
    
    #knotpos = (2950, 1976) # lower knot
    knotpos = (2981, 2006) # upper knot
    
    rmsfile = 'hlsp_relics_hst_wfc3ir-60mas_whl0137-08_f110w_rms.fits'
    rmsfile = os.path.join(indir, rmsfile)
    argdict = initArgDict(rmsfile, imstamp, limits=(xlo,xhi,ylo,yhi), ax=ax, ay=ay,
                          knotpos=knotpos, sourcegrid=(xss,yss), star=star)
    
    theta = np.array([.001, pc_to_pix(450, 6.2)])
    sampler = runMCMC(theta, argdict, niter=5000, outfile='topknot_5k_4walk.h5')
    
    return sampler


def initDeflection_Image(imagefile, deflectionFileX, deflectionFileY, 
                         zlens, zsource_in, zsource_out):
    
    ### Read in image file
    ### Read in deflection maps, 
    ### reproject to image file coordinates
    
    imhdu = fits.open(imagefile)
    imdata = imhdu[0].data
    
    hdudeflx = fits.open(deflectionFileX)
    hdudefly = fits.open(deflectionFileY)
    
    #reproject deflection fields to HST WCS pixels
    deflxHST, footprint = reproject_interp(hdudeflx[0], imhdu[0].header)
    deflyHST, footprint = reproject_interp(hdudefly[0], imhdu[0].header)
    ax = deflxHST / 0.06 # arcsec -> pixels
    ay = deflyHST / 0.06 # arcsec -> pixels
    
    # convert back to Dds / Ds = 1
    Dds_Ds_in  = Dds_Ds(zlens, zsource_in)
    Dds_Ds_out = Dds_Ds(zlens, zsource_out)

    ax = ax / Dds_Ds_in * Dds_Ds_out
    ay = ay / Dds_Ds_in * Dds_Ds_out
    
    return ax, ay, imdata


def Dds_Ds(zl, zs):
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    Dds = cosmo.angular_diameter_distance_z1z2(zl, zs)
    Ds  = cosmo.angular_diameter_distance_z1z2(0 , zs)
    return Dds / Ds


def makeMagnificationMap(ax, ay, outfile=None):
    axx = ddx(ax)
    ayy = ddy(ay)
    
    axy = ddy(ax)
    ayx = ddx(ay)
    
    kappa  = 0.5 * (axx + ayy)
    gamma1 = 0.5 * (axx - ayy)
    gamma2 = axy
    gamma  = np.sqrt(gamma1**2 + gamma2**2)
    
    kappa = zeropad(kappa)
    gamma = zeropad(gamma)
    
    magnifinv = (1-kappa)**2 - gamma**2
    
    if outfile:
        magnif = 1. / magnifinv
        hdumag = fits.PrimaryHDU(magnif)
        hdumag.header = imhdu[0].header
        hdumag.writeto(outfile)
    
    return magnifinv


def ddy(A):
    """Numerical derivative: 2nd-order
    output array will have dimentions (N-2, N-2)"""
    dAdy1 = (A[2:] - A[:-2]) / 2.
    dAdy2 = (-A[4:] + 8*A[3:-1] - 8*A[1:-3] + A[:-4]) / 12.
    dAdy1[1:-1,:] = dAdy2
    dAdy1 = dAdy1[:,1:-1]
    return dAdy1

def ddx(A):
    dAdx = ddy(A.T).T
    return dAdx

# Results have dimensions (N-2, N-2)
# Add zero padding to restore to (N, N) HST image pixel grid
def zeropad(data):
    ny, nx = data.shape
    padded = np.zeros((ny+2, nx+2))
    padded[1:-1,1:-1] = data
    return padded

def starGen(imdata, starLoc, extent=15):
    xmin, xmax = (starLoc[0]-(extent/2), starLoc[0]+((extent/2)))
    xmin, xmax = int(xmin), int(xmax)
    ymin, ymax = (starLoc[1]-(extent/2), starLoc[1]+((extent/2)))
    ymin, ymax = int(ymin), int(ymax)
    star = imdata[ymin:ymax, xmin:xmax]
    return star


def initArgDict(rmsfile, imstamp, limits, ax, ay,
                knotpos, sourcegrid, star, delta=5):
    xlo, xhi, ylo, yhi = limits
    x, y = knotpos
    xs = x - ax[y, x]
    ys = y - ay[y, x]
    xss, yss = sourcegrid
    rmsfo = fits.open(rmsfile)
    rms = rmsfo[0].data
    rmscut = rms[ylo:yhi,xlo:xhi]
    
    datax, datay = x - xlo, y - ylo
    
    rms_cutout = rmscut[datay-delta:datay+delta, datax-delta:datax+delta]
    knotbounds = [datax-delta, datax+delta, datay-delta,datay+delta]
    # Include in args: xs, ys, xss, yss, star, data, sigma=RMS
    argdict = {
        "xs" : xs,
        "ys" : ys,
        "xss" : xss,
        "yss" : yss,
        "star" : star,
        "arcIm" : imstamp,
        "sigma" : rms_cutout,
        "knotbounds" : knotbounds
    }
    return argdict


def runMCMC(theta_init, argdict, nwalkers=4, niter=500, outfile=None, **kwargs):
    ndim = len(theta_init)
    pos = theta_init + 1e-5 * np.random.randn(nwalkers, ndim)
    
    if outfile:
        backend = emcee.backends.HDFBackend(outfile)
        backend.reset(nwalkers,ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, prob, kwargs=argdict, backend=backend)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, prob, kwargs=argdict)
    
    sampler.run_mcmc(pos, niter, progress=True)
    return sampler


def prob(theta, **kwargs):
    pri = prior(theta)
    if not np.isfinite(pri):
        return -np.inf
    return pri + log_likelihood(theta,**kwargs)
    

def prior(theta):
    amp, reff = theta
    if 0 < amp < 10 and 0 < reff < 50:
        return 0
    return -np.inf
    

def log_likelihood(theta, **kwargs):
    chisq = chisquared(theta, **kwargs)
    #print(chisq)
    sigma = kwargs["sigma"]
    lhood = (1. / (sigma * np.sqrt(2.*np.pi))) * np.exp(-chisq / 2.)
    #print(lhood)
    log_lhood = np.log(lhood)
    #print(log_lhood)
    result = np.sum(log_lhood)
    return result


def chisquared(theta, **kwargs):
    sigma = kwargs["sigma"]
    arcIm = kwargs["arcIm"]
    simIm = np.zeros_like(arcIm)
    conv = convolved(theta, **kwargs)
    simIm[:,:] += conv[:,:] # add sim image to data image
    # cut out each knot
    knotbounds = kwargs["knotbounds"]
    trueKnot = arcIm[knotbounds[2]:knotbounds[3], knotbounds[0]:knotbounds[1]]
    simKnot = simIm[knotbounds[2]:knotbounds[3], knotbounds[0]:knotbounds[1]]
    # chisquared calculation
    result = (1/sigma**2) * (trueKnot - simKnot)**2
    return result


def convolved(theta, **kwargs): 
    amp, reff = theta
    #amp, reff = preconv(theta)
    #print(amp,reff)
    xs, ys = kwargs["xs"], kwargs["ys"]
    xss, yss = kwargs["xss"], kwargs["yss"]
    star = kwargs["star"]
    sersic = Sersic2D(amplitude=amp, r_eff=reff, n=4, x_0=xs, y_0=ys)
    S1 = sersic(xss, yss).value
    xlo, xhi = 2700, 3100
    ylo, yhi = 1800, 2100
    S1stamp = S1[ylo:yhi, xlo:xhi]
    S1conv = convolve(S1stamp, star)
    return S1conv


def preconv(theta):
    #This needs to be fixed...
    flux, rpc = theta
    rpix = pc_to_pix(rpc, 6.2)
    dist = cosmo.angular_diameter_distance_z1z2(0 , 6.2).value * 10**6 #returns in pc
    print(dist)
    magnif = 300 #??
    coef = (22.665 * magnif) / (4 * np.pi * dist**2)
    print(coef)
    amp = flux / (coef * rpc**2)
    return amp, rpix


def pc_to_pix(r, z):
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    DA = cosmo.angular_diameter_distance_z1z2(0 , z).value * 10**6 #returns in pc
    theta = r / DA # radian
    theta = theta * (180 / np.pi) * 3600 # arcsec
    pixSize = 0.06 #HST WFC3 pixel size
    rpix = theta / pixSize
    return rpix


def pix_to_pc(rpix, z):
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    DA = cosmo.angular_diameter_distance_z1z2(0 , z).value * 10**6 #returns in pc
    pixSize = 0.06 #HST WFC3 pixel size
    theta = rpix * pixSize #arcsec
    theta = theta / ((180. / np.pi) * 3600) #radian
    r = theta * DA #phys
    return r

