# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 11:41:55 2021

common functions

@author: Martin
"""

import numpy as  np
from nptdms import TdmsFile
from skimage.restoration import unwrap_phase
from scipy.optimize import curve_fit
from scipy.interpolate import interp2d

#%%
def get_images(path, pixelside):

    """
    Parameters
    ----------
    path : r+path
        leads to a tdms file containing 3d-data (2d image + 1d different images)
    pixelside : TYPE
        pixels per side

    Returns
    -------
    r1 : TYPE
        3d matrix containing images
    num : TYPE
        num images
    """
   
    tdms_file = TdmsFile.read(path)
    raw = tdms_file['Untitled']
    raw_c = raw['Untitled']
    num = int(raw_c[:].shape[0] / pixelside)
    r1 = np.zeros(shape=(540, num, 540))
    r1[0,:,:] = raw_c[:].reshape(num, 540)

    for i in range(539):
        raw_c = raw['Untitled ' +str(i+1)]
        r1[i+1,:,:] = raw_c[:].reshape(num, 540) 
    
    r1 = np.transpose(r1, (0,2,1))[:,:,0]
    
    return r1


def extractHologram(path, ref_pump, ref_probe, x_pump, y_pump, x_probe, y_probe, rad, cropShiftPad, divideByRefs):
    """
    Parameters
    ----------
    path: r + path
    ref_pump: reference for which pump is divided
    ref_probe ... 
    x,y,... points and radius of fft selection
    cropAndShift: boolean if the new image dimension should match the former one and fftshift is applied before back-trafo

    Returns
    -------
    amplitudes and phases of pump and probe
    """
    
    im = get_images(path, 540)
    
    probe, pump = getOffAxis(im,  x_pump, y_pump, x_probe, y_probe, rad, cropShiftPad)
    
    
    
    if divideByRefs:
        if cropShiftPad:
            x = np.arange(0, ref_pump.shape[0])
            y = np.arange(0, ref_probe.shape[1])
            f1 = interp2d(x, y, ref_pump, kind='cubic')
            f2 = interp2d(x, y, ref_probe, kind='cubic')
            ref_pump = f1(np.arange(0, pump.shape[0]), np.arange(0, pump.shape[1]))
            ref_probe = f2(np.arange(0, pump.shape[0]), np.arange(0, pump.shape[1]))
        probe_amplitude = np.abs(probe) / np.sqrt(ref_probe)
        pump_amplitude = np.abs(pump) / np.sqrt(ref_pump)
    else:
        probe_amplitude = np.abs(probe)
        pump_amplitude = np.abs(pump)
    
    pump_phase = np.angle(pump)
    probe_phase = np.angle(probe)
    
    pump  = np.multiply(pump_amplitude, np.exp(1j *pump_phase))
    probe = np.multiply(probe_amplitude, np.exp(1j * probe_phase))
    return pump, probe


def extractHologram01(path, dark, x_pump, y_pump, x_probe, y_probe, rad, cropShiftPad, divideByRefs, *refs):
    """
    Parameters
    ----------
    path: r + path
    ref_pump: reference for which pump is divided
    ref_probe ... 
    x,y,... points and radius of fft selection
    cropAndShift: boolean if the new image dimension should match the former one and fftshift is applied before back-trafo

    Returns
    -------
    amplitudes and phases of pump and probe
    """
    
    im = get_images(path, 540)
    
    probe, pump = getOffAxis(im - dark,  x_pump, y_pump, x_probe, y_probe, rad, cropShiftPad)
    
    
    
    if divideByRefs:
        if cropShiftPad:
            x = np.arange(0, refs[0].shape[0])
            y = np.arange(0, refs[0].shape[1])
            f1 = interp2d(x, y, refs[0], kind='cubic')
            f2 = interp2d(x, y, refs[1], kind='cubic')
            ref_pump = f1(np.arange(0, pump.shape[0]), np.arange(0, pump.shape[1]))
            ref_probe = f2(np.arange(0, pump.shape[0]), np.arange(0, pump.shape[1]))
        probe_amplitude = np.abs(probe) / np.sqrt(ref_pump)
        pump_amplitude = np.abs(pump) / np.sqrt(ref_probe)
    else:
        probe_amplitude = np.abs(probe)
        pump_amplitude = np.abs(pump)
    
    pump_phase = np.angle(pump)
    probe_phase = np.angle(probe)
    
    pump  = np.multiply(pump_amplitude, np.exp(1j *pump_phase))
    probe = np.multiply(probe_amplitude, np.exp(1j * probe_phase))
    return pump, probe


def linOffset(x, a, b, offset):
    X = x[0]
    Y = x[1]
    G =  (X * a + Y * b ) + offset
    return G.ravel()

def zernikes(x, a, b, c, d, e, f):
    X = x[0]
    Y = x[1]
    #G =  (X * a + Y * b ) + offset
    G = a * zernike(1, X,Y) + b * zernike(2, X,Y) + c * zernike(3, X,Y) + d * zernike(4, X, Y) * e * zernike(5, X,Y) + f * zernike(6, X,Y)
    return G.ravel()

def zernikes3(x, a, b, c):
    X = x[0]
    Y = x[1]
    #G =  (X * a + Y * b ) + offset
    G = a * zernike(1, X,Y) + b * zernike(2, X,Y) + c * zernike(3, X,Y) 
    return G.ravel()

def cart_to_pol(x, y, x_c = 0, y_c = 0, deg = True):
    complex_format = x - x_c + 1j * (y - y_c)
    return np.abs(complex_format), np.angle(complex_format)

def zernike(n, xv, yv):
    rho, phi = cart_to_pol(xv, yv)
    
    if n==1:
        return  np.ones(rho.shape)
    if n==2:
        return 2 * np.multiply(rho, np.sin(phi))
    if n==3:
        return 2 * np.multiply(rho, np.cos(phi))
    if n==4:
        return np.sqrt(6) * rho**2 * np.sin(2*phi)
    if n==5:
        return np.sqrt(3) * (2*rho**2 - 1)
    if n==6:
        return np.sqrt(6) * rho**2 * np.cos(2*phi)

def getPhase(input_phase, meshgridx, meshgridy):
    """
    Parameters
    ----------
    input_phase : TYPE
        DESCRIPTION.
    meshgridx : TYPE
        DESCRIPTION.
    meshgridy : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        subtracted phase as well as subtracted portion.

    """
    x = np.vstack((meshgridx.ravel(), meshgridy.ravel()))
    
    u_phase = np.ravel(unwrap_phase(input_phase))
    
    #boundss = ([-1e3, -10, -10, -1, -1, -1], [1e3, 10, 10, 1, 1, 1])
    #tic = time.perf_counter()
    popt, dd = curve_fit(zernikes, x, u_phase)#,ftol=0.1, xtol=0.1)
    #toc = time.perf_counter()
    #print(toc-tic)
    return (u_phase - zernikes(x, *popt)).reshape(meshgridx.shape[0], meshgridy.shape[1]), zernikes(x, *popt).reshape(meshgridx.shape[0], meshgridy.shape[1])

def getOffAxis(hologram, x_pump, y_pump, x_probe, y_probe, rad, cropShiftPad):
    pu = np.zeros(hologram.shape, dtype = np.complex_)
    pr = np.zeros(hologram.shape, dtype = np.complex_)
    
    norm = 'forward'
    
    fft = np.fft.fftshift(np.fft.fft2(hologram, norm=norm))
    
    for x in range(pu.shape[0]):
        for y in range(pu.shape[1]):
            if (x_pump - x)**2 + (y_pump - y)**2 < rad**2:
                pu[x,y] = fft[x,y]
            if (x_probe - x)**2 + (y_probe - y)**2 < rad**2:
                pr[x,y] = fft[x,y]
    
    if cropShiftPad:
        pu = pu[x_pump - rad : x_pump + rad, y_pump - rad : y_pump + rad]
        pr = pr[x_probe - rad : x_probe + rad, y_probe - rad : y_probe + rad]
        
        # zero binning
        size = 256
        pu2 = np.zeros(shape=(size, size), dtype = complex)
        pr2 = np.copy(pu2)
        pu2[int(size/2) - rad : int(size/2) + rad, int(size/2) - rad : int(size/2) + rad] = pu
        pr2[int(size/2) - rad : int(size/2) + rad, int(size/2) - rad : int(size/2) + rad] = pr
        
        pu = np.fft.fftshift(pu2)
        pr = np.fft.fftshift(pr2)
            
    pump = np.fft.ifft2(pu, norm = norm)
    probe = np.fft.ifft2(pr, norm = norm)
    # hologram2 = np.fft.ifft2(fft, norm = norm)
    
    return pump, probe

def angularSpectrumMethod(probe, numPix, spacPix, lam, z):
    """
    

    Parameters
    ----------
    probe : TYPE
        amplitude * exp(i phase)
    numPix : TYPE
        number of pixels
    spacPix : TYPE
        dimension of each pixel
    lam : TYPE
        DESCRIPTION.
    z : TYPE
        propagation distance

    Returns
    -------
    propagated : TYPE
        DESCRIPTION.

    """
    f = 1/numPix/spacPix
    c = np.arange(1, numPix)
    C,R = np.meshgrid(c, c)
    
    probe_fft = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(probe)))
    SPTF = np.exp(1j * 2 * np.pi * z * (lam**(-2) - ((R - numPix/2 - 1) * f)**2 - ((C - numPix/2 - 1) * f)**2)**(0.5))
    propagated = np.fft.fftshift((np.fft.fft2(np.fft.fftshift(np.multiply(probe_fft,SPTF)))))
    return propagated

# M = pos.shape[0] #num of pixels
# dx = 6.9e-6
# lam = 525e-09

#%%
def autofocus(image, numPix, spacePix, lam, zrange, samples, iterations):
    
    E = np.zeros(samples)
    firstE = np.zeros(samples)
    firstz = np.zeros(samples)
    zs = np.zeros(iterations)
    zstart = 0
    zmin = 0
    
    for ii in range(iterations):
        print('iteration:')
        print(ii)
        
        val = 0.25 * 2**(ii)
        z = np.linspace(zstart-zrange/val,zstart + zrange/val, samples)
        print('z:')
        print(zstart)
        
        for zz in range(samples):
            E[zz] = np.sum(np.abs(angularSpectrumMethod(image, numPix, spacePix, lam, z[zz])))
            
        zmin = np.argmin(E)     
        
        zstart = z[zmin]
        zs[ii] = zstart
        
        if ii == 0:
            firstz = z + 0
            firstE = E + 0
            print('argmin:, zstart:')
            print(np.argmin(E) , zstart)
        # print(zmin, zstart)
        
    return z[zmin], zs