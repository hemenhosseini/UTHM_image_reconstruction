# -*- coding: utf-8 -*-
"""
Created on Fri May  7 16:57:14 2021

@author: Martin
"""

from nptdms import TdmsFile
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import common_functions as cf
import os
import numpy as np
from skimage.restoration import unwrap_phase
from skimage.feature import peak_local_max
import time
from scipy.signal import find_peaks

#%% first figure out the positions of the cc terms (e.g. donuts), redo until you found the right indices

# load any hologram and perform fft
pos = cf.get_images(r"C:\Users\Desktop\Documents\UHTMicroscopy\data\2021-06-11\690nm\run2\run\run0delay0.tdms", 540)

posFFT = np.fft.fftshift(np.fft.fft2(pos, norm = 'forward'))

# indices and radius (try out)
y_pump =88
x_pump = 85
y_probe = 87
x_probe = 450

rad = 80

# cut out cc terms
pu = np.zeros(pos.shape, dtype = np.complex_)
pr = np.zeros(pos.shape, dtype = np.complex_)

for x in range(pu.shape[0]):
    for y in range(pu.shape[1]):
        if (x_pump - x)**2 + (y_pump - y)**2 < rad**2:
            pu[x,y] = posFFT[x,y]
        if (x_probe - x)**2 + (y_probe - y)**2 < rad**2:
            pr[x,y] = posFFT[x,y]

# if the contrast is bad change clim settings
plt.figure(110)
h = plt.imshow(np.abs(pu + pr), origin='lower', cmap='turbo')
h.set_clim([0,0.05])
# plt.colorbar()
plt.ylabel('x')
plt.xlabel('y')
plt.title('pump selection (uncropped)')

#%% if you want to divide the amplitudes by the references (matz didnt)

ref1 = cf.get_images(r"C:\Users\Desktop\Documents\UHTMicroscopy\data\2021-06-11\690nm\run2\ref2.tdms", 540)
ref2 = cf.get_images(r"C:\Users\Desktop\Documents\UHTMicroscopy\data\2021-06-11\690nm\run2\ref1.tdms", 540)


#%% extract from files, enter number of runs and delays as well as the path to files 

num_runs = 17
num_delays = 130
data = np.ndarray((num_runs, num_delays, 2, 256, 256), dtype = 'complex_')

# load dark image
dark = cf.get_images(r"C:\Users\Desktop\Documents\UHTMicroscopy\data\2021-06-11\690nm\run2\dark.tdms", 540)
cwdd = r'C:\Users\Desktop\Documents\UHTMicroscopy\data\2021-06-11\690nm\run2\run'
os.chdir(cwdd)
    

files = os.listdir(cwdd)
# i = 1
for i, filename in enumerate(files):
    # ext = filename[0:24]
    # filename = filename[24:]
    if filename[0:3] != 'pos' and filename[-5:-1] != "inde" and filename[-8:-1] != 'dms.tdm' and (filename[-4:-1] == 'tdm'):
        
        path = cwdd + "\\"  + filename
        # print(path)
        # pump, probe = cf.extractHologram01(path, x_pump, y_pump, x_probe, y_probe, rad, True, True, ref1, ref2)
        pump, probe = cf.extractHologram01(path, dark, x_pump, y_pump, x_probe, y_probe, rad, True, False)
        
        f = filename[0:-5]
        if f[4].isdigit():
            run = int(f[3:5])
            delay = int(f[10:])
        else:
            run = int(f[3])
            delay = int(f[9:])
        
        if run%2 != 0:
            delay = num_delays - delay - 1
    
        data[run, delay, 0, :,:] = pump
        data[run, delay, 1, :,:] = probe

        print(str("{0:.3f}".format(i/(num_runs * num_delays))))
        
        # i = i + 1


np.save("data", data)




























#%%

ref1 = cf.get_images(r"C:\Users\Desktop\Documents\UHTMicroscopy\data\2021-06-11\690nm\run2\ref2.tdms", 540)
ref2 = cf.get_images(r"C:\Users\Desktop\Documents\UHTMicroscopy\data\2021-06-11\690nm\run2\ref1.tdms", 540)

pump_pos, probe_pos = cf.extractHologram(r"C:\Users\Desktop\Documents\UHTMicroscopy\data\2021-06-11\690nm\run2\run\run0delay69.tdms", ref1, ref2, x_pump, y_pump, x_probe, y_probe, rad, True, False)
pump_neg, probe_neg = cf.extractHologram(r"C:\Users\Desktop\Documents\UHTMicroscopy\data\2021-06-11\690nm\run2\run\run0delay0.tdms", ref1, ref2, x_pump, y_pump, x_probe, y_probe, rad, True, False)

posdif = np.abs(pump_pos) - np.abs(probe_pos)
negdif = np.abs(pump_neg) - np.abs(probe_neg)

#%% static image

probe = cf.get_images(r"C:\Users\Desktop\Documents\UHTMicroscopy\data\2021-06-09\.tdms", 540)

plt.figure(11)
a = plt.imshow(probe, origin = 'lower')#, vmin = 0, vmax = 25)
plt.colorbar(a)
plt.title('probe beam only')

#%% check refs

#get max and min values of refs

maxx = np.max((ref1, ref2))
minn = np.min((ref1, ref2))

fig = plt.figure(1)
gs = fig.add_gridspec(1, 3, hspace=0.1, wspace=0.1)
(ax1, ax2, ax3) = gs.subplots(sharex='col', sharey='row')
fig.suptitle('References')
im1 = ax1.imshow(ref1, origin = 'lower', vmin=minn, vmax=maxx)
# fig.colorbar(im, ax = ax1)
im2 = ax2.imshow(ref2, origin = 'lower', vmin=minn, vmax=maxx)
im3 = ax3.imshow(ref2-ref1, origin = 'lower', cmap = 'PiYG', vmin = -50, vmax = 50, label='ref2-ref1')
plt.suptitle('ref2-ref1')
# fig.colorbar(im3, ax = (ax1,ax2, ax3))

fig.colorbar(im2, ax = (ax1,ax2))
fig.colorbar(im3, ax = ax3)


ax1.title.set_text('ref1')
ax2.title.set_text('ref2')
ax3.title.set_text('ref2-ref1')

#%%
maxx = np.max((np.abs(pump_pos), np.abs(probe_pos), np.abs(probe_neg), np.abs(pump_neg)))
minn = np.min((np.abs(pump_pos), np.abs(probe_pos), np.abs(probe_neg), np.abs(pump_neg)))

contr = 1

difpos = np.abs(pump_pos)-np.abs(probe_pos)
difneg = np.abs(pump_neg)-np.abs(probe_neg)

dSSpos = np.divide(difpos, np.abs(pump_pos))
dSSneg = np.divide(difneg, np.abs(pump_neg))

fig = plt.figure(2)
fig.clear(True)
gs = fig.add_gridspec(2, 3, hspace=0.1, wspace=0.1)
(ax1, ax2, ax3), (ax4, ax5, ax6) = gs.subplots(sharex='col', sharey='row')
im1 = ax1.imshow(np.abs(pump_pos), origin = 'lower')#, vmin=minn, vmax=maxx)
ax1.title.set_text('pos pump')
im2 = ax2.imshow(np.abs(probe_pos), origin = 'lower')#, vmin=minn, vmax=maxx)
im3 = ax3.imshow(dSSpos, origin = 'lower', cmap = 'PiYG', vmin = -contr, vmax = contr, label='dif. pos')

im4 = ax4.imshow(np.abs(pump_neg), origin = 'lower')#, vmin=minn, vmax=maxx)
ax4.title.set_text('neg pump')
im5 = ax5.imshow(np.abs(probe_neg), origin = 'lower')#, vmin=minn, vmax=maxx)
im6 = ax6.imshow(dSSneg, origin = 'lower', cmap = 'PiYG', vmin = -contr, vmax = contr, label='dif. pos')
# fig.colorbar(im3, ax = (ax1,ax2, ax3))

fig.colorbar(im1, ax = ax1)
fig.colorbar(im2, ax = ax2)
fig.colorbar(im3, ax = ax3)
fig.colorbar(im4, ax = ax4)
fig.colorbar(im5, ax = ax5)
fig.colorbar(im6, ax = ax6)

#%%
contr = 1
fix, ax = plt.subplots()
im = plt.imshow(posdif - negdif, origin = 'lower', cmap = 'PiYG', vmin = -contr, vmax = contr)
ax.title.set_text('difpos - difneg')
plt.colorbar(im, ax = ax)

#%% extract from files  

num_runs = 20
num_delays = 20
data = np.ndarray((num_runs, num_delays, 2, 256, 256), dtype = 'complex_')

cwdd = r'C:\Users\Desktop\Documents\UHTMicroscopy\data\2021-06-15 Renzo Cell\position3\run'
os.chdir(cwdd)
    
files = os.listdir(cwdd)
# print(files)
i = 1
for filename in files:
    # ext = filename[0:24]
    # filename = filename[24:]
    if filename[0:3] != 'pos' and filename[-5:-1] != "inde" and filename[-8:-1] != 'dms.tdm' and (filename[-4:-1] == 'tdm'):
        
        path = cwdd + "\\"  + filename
        # print(path)
        pump, probe = cf.extractHologram(path, ref1, ref2, x_pump, y_pump, x_probe, y_probe, rad, True, False)
        
        f = filename[0:-5]
        if f[4].isdigit():
            run = int(f[3:5])
            delay = int(f[10:])
        else:
            run = int(f[3])
            delay = int(f[9:])
        
        if run%2 != 0:
            delay = num_delays - delay - 1
    
        data[run, delay, 0, :,:] = pump
        data[run, delay, 1, :,:] = probe

        print(str("{0:.3f}".format(i/(num_runs * num_delays))))
        
        i = i + 1


np.save("data", data)

#%%
data = np.load(r"C:\Users\Desktop\Documents\UHTMicroscopy\data\2021-05-07 TEA2 PBI4\Analysis\data.npy")

#%%
tdms_file = TdmsFile.read(r"C:\Users\Desktop\Documents\UHTMicroscopy\data\2021-06-15 Renzo Cell\run\image2withref1_darkfieldpos.tdms")

raw = tdms_file['Untitled']
raw = raw['Untitled']
pos = raw[:]

times = 4*(pos - pos[0])*1e-3 / 2.9979e8 # double pass, pos in mm
times = times * 1e12 # ps

del(raw, tdms_file)


#%% pump probe signal

pumpProbe = np.zeros(shape = (num_runs, pos.shape[0]))

for ii in range(num_runs):
    
    pumpMinusProbe = (np.abs(data[ii, :, 1, :, :]) - np.abs(data[ii, :, 0, :, :]))
    
    pumpMinusProbe = np.mean(pumpMinusProbe, axis=(1,2))    
    
    pumpProbe[ii,:] = pumpMinusProbe


#pumpMinusProbe = np.mean(pumpMinusProbe, axis=(2,3))

ind = np.arange(0,times.shape[0],1)

fig, (ax1, ax2) = plt.subplots(2,1)
# # ax1.plot(ind, pumpMinusProbe[0], ind, pumpMinusProbe[1], ind, pumpMinusProbe[2], ind, pumpMinusProbe[3])
for ii in range(num_runs):
    ax1.plot(times, pumpProbe[ii])
ax1.set_xlabel('time (ps)')
ax1.set_ylabel('mean (pump - probe) (a.u.)')
ax1.legend(['1', '2', '3', '4'])
ax1.grid()

ax2.plot(pos, pumpProbe[0], '-o')
ax2.set_xlabel('index')
ax2.set_ylabel('mean (pump - probe) (a.u.)')
ax2.legend(['1', '2', '3', '4'])
ax2.grid()

#%% images pump minus probe
contr = 1

for i in range(0, pos.shape[0], 1):
# i = 17
    print(i)
    im = np.abs(data[:, i, 1, :, :]) - np.abs(data[:, i, 0, :, :])
    bg = np.abs(data[:, 0, 1, : :,]) - np.abs(data[:, 0, 0, :, :])
    
    im = im - bg
    # im = np.angle(data[:, i, 1, :, :])# -data[:, i, 0, :, :])
    # im = np.divide(im, np.abs(data[:, i, 0, :, :]))
    im = np.mean(im, axis=0)
    fig = plt.figure(2)
    im = plt.imshow(im, origin = 'lower', cmap = 'PiYG', vmin = -contr, vmax = contr)
    plt.colorbar()
    # im.set_clim(-100, 100)
    plt.title('time: ' + '{:.1f}'.format(-times[5]+times[i]) + 'ps')
    plt.savefig(str(i)+'.png')
    fig.clear(True)
    
#%%  



contr = 20

for i in range(0, pos.shape[0], 3):
# i = 17
    print(i)
    im = np.real(data[:, i, 1, :, :]) - np.real(data[:, i, 0, :, :])
    bg = np.real(data[:, 0, 1, : :,]) - np.real(data[:, 0, 0, :, :])
    
    im = im - bg
    # im = np.angle(data[:, i, 1, :, :])# -data[:, i, 0, :, :])
    # im = np.divide(im, np.abs(data[:, i, 0, :, :]))
    im = np.sum(im, axis=0)
    fig = plt.figure(2)
    im = plt.imshow(im, origin = 'lower', cmap = 'PiYG', vmin = -contr, vmax = contr)
    plt.colorbar()
    # im.set_clim(-100, 100)
    plt.title('time: ' + '{:.1f}'.format(-times[22]+times[i]) + 'ps')
    plt.savefig(str(i)+'.png')
    fig.clear(True)
    
    
#%% find coordinates of pump spots
i = 36
im = np.abs(data[:, i, 1, :, :]) - np.abs(data[:, i, 0, :, :])
bg = np.abs(data[:, 0, 1, : :,]) - np.abs(data[:, 0, 0, :, :])

im = im - bg
# im = np.angle(data[:, i, 1, :, :])# -data[:, i, 0, :, :])
# im = np.divide(im, np.abs(data[:, i, 0, :, :]))
im = np.abs(np.sum(im, axis=0))

# im = np.abs(data[1,0,0,:,:])
#im = np.abs(posdif - negdif)
coordinates = peak_local_max(im, min_distance=8, threshold_rel=0.05)

# display results
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax1, ax2 = ax.ravel()
ax1.imshow(im, cmap=plt.cm.gray)
plt.xlabel('x')
# ax1.axis('off')
ax1.set_title('Original')

ax2.imshow(im, cmap=plt.cm.gray)
ax2.autoscale(False)
ax2.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
ax2.axis('off')
ax2.set_title('Peak local max')

fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                    bottom=0.02, left=0.02, right=0.98)

plt.show()
print(coordinates.shape[0])

#%%
# devv = np.std(cosig, axis = 0)

peaks, _ = find_peaks(np.abs(cosig[22,:]), height =30)#rel_height=0.5)

c2 = coordinates[peaks,:]


plt.figure(10)
plt.plot(devv)


#%% individual particles

radius = 4

cosig = np.zeros(shape=(pos.shape[0], coordinates.shape[0]))

i = 0

for (x,y) in coordinates:
    
    sig = np.abs(data[:,:, 1, x-radius:x+radius, y-radius:y+radius]) - np.abs(data[:,:, 0, x-radius:x+radius, y-radius:y+radius])
    bg = np.abs(data[:,0, 1, x-radius:x+radius, y-radius:y+radius]) - np.abs(data[:,0, 0, x-radius:x+radius, y-radius:y+radius])
    sig = sig - bg[:,None,:,:]## sig = np.divide(np.abs(data[:,:, 1,  x-radius:x+radius, y-radius:y+radius]) - np.abs(data[:,:, 0,  x-radius:x+radius, y-radius:y+radius]), np.abs(data[:,:, 1,  x-radius:x+radius, y-radius:y+radius]))  
    cosig[:,i] = np.sum(sig, axis=(0, 2, 3)) 
    # cosig[:,i] = np.mean(sig, axis=(0))
    i += 1
    
ind1 = 0
ind2 = 100

fig, ax = plt.subplots()
# ax.plot(times[ind1:ind2], cosig[ind1:ind2,:], '-o', markersize = 0.3, linewidth = 1)
ax.plot(times, cosig, '-o', markersize = 0.3, linewidth = 1)
# ax.plot(np.arange(ind1, ind2, 1), cosig[ind1:ind2,:], '-o', markersize = 0.3, linewidth = 1)
ax.legend()
ax.set_xlabel('time(ps)')

#%% individual particles 2

radius = 3

cosig = np.zeros(shape=(pos.shape[0], c2.shape[0]))

i = 0

for (x,y) in c2:
    
    sig = np.abs(data[:,:, 1, x-radius:x+radius, y-radius:y+radius]) - np.abs(data[:,:, 0, x-radius:x+radius, y-radius:y+radius])
    bg = np.abs(data[:,0, 1, x-radius:x+radius, y-radius:y+radius]) - np.abs(data[:,0, 0, x-radius:x+radius, y-radius:y+radius])
    sig = sig - bg[:,None,:,:]## sig = np.divide(np.abs(data[:,:, 1,  x-radius:x+radius, y-radius:y+radius]) - np.abs(data[:,:, 0,  x-radius:x+radius, y-radius:y+radius]), np.abs(data[:,:, 1,  x-radius:x+radius, y-radius:y+radius]))  
    cosig[:,i] = np.sum(sig, axis=(0, 2, 3)) 
    # cosig[:,i] = np.mean(sig, axis=(0))
    i += 1
    
ind1 = 0
ind2 = 100

fig, ax = plt.subplots()
ax.plot(times[ind1:ind2], cosig[ind1:ind2,:], '-o', markersize = 0.3, linewidth = 1)
ax.legend()
ax.set_xlabel('time(ps)')

#%% dS or dS/S

i = 24

#dS image
im = np.abs(data[:, i, 1, :, :]) - np.abs(data[:, i, 0, :, :])
bg = np.abs(data[:, 0, 1, : :,]) - np.abs(data[:, 0, 0, :, :])
im = im - bg
dS = np.sum(im, axis=0)

#dS/S (I dont know which is on and off...)
dSS = np.divide(im, np.abs(data[:, i, 1, :, :]))
dSS = np.sum(dSS, axis=0)

fig, (ax1, ax2) = plt.subplots(1, 2)
im1 = ax1.imshow(dS, origin = 'lower')
plt.colorbar(im1, ax = ax1)
contr = 10
im2 = ax2.imshow(dSS, origin = 'lower', vmin=-contr, vmax = contr)
plt.colorbar(im2, ax = ax2)#

#%% images for single runs

contr = 1
i = 24

for ii in range(num_runs):
# i = 17
    print(ii)
    im = np.abs(data[ii, i, 1, :, :]) - np.abs(data[ii, i, 0, :, :])
    bg = np.abs(data[ii, 0:20, 1, : :,]) - np.abs(data[ii, 0:20, 0, :, :])
    bg = np.mean(bg, axis = 0)
    im = im - bg
    # im = np.angle(data[:, i, 1, :, :])# -data[:, i, 0, :, :])
    # im = np.divide(im, np.abs(data[:, i, 0, :, :]))
    # im = np.sum(im, axis=0)
    fig = plt.figure(2)
    im = plt.imshow(im, origin = 'lower', cmap = 'PiYG', vmin = -contr, vmax = contr)
    plt.colorbar()
    # im.set_clim(-100, 100)
    plt.title('run: ' + str(ii))
    plt.savefig(str(ii)+'.png')
    fig.clear(True)
    
    
#%% 540x540 image
cwdd = r"C:\Users\Martin\Documents\UHT Microscopy\data\2021-06-21 Renzo Sample 8\run"

imm = np.zeros(shape=(num_runs, 2, 540, 540), dtype = 'complex_')
bg = np.copy(imm)
for i in range(num_runs):
    
    path = cwdd + "\\run" + str(i) + "delay24.tdms"
    pathbg = cwdd + "\\run" + str(i) + "delay0.tdms"
    t1 = time.time()
    pump, probe = cf.extractHologram(path, ref1, ref2, x_pump, y_pump, x_probe, y_probe, rad, False, False)
    pumpbg, probebg = cf.extractHologram(pathbg, ref1, ref2, x_pump, y_pump, x_probe, y_probe, rad, False, False)
    t2 = t1 - time.time()
    print('extraction')
    print(t2)
    
    t3 = time.time()
    imm[i, 0, :, :] = pump
    imm[i, 1, :, :] = probe
    bg[i, 0, :, :] = pumpbg
    bg[i, 1, :, :] = probebg
    t4 = t3 - time.time()
    print(t4)
    
#%% plot 540x540 images

im = np.angle(imm[:,1,:,:]) - np.angle(imm[:,0,:,:])
bgg = np.angle(bg[:,1,:,:]) - np.angle(bg[:,0,:,:])
im = im - bgg
im = np.sum(im, axis = 0)

fig, ax = plt.subplots()
a = ax.imshow(im, origin = 'lower', cmap = 'twilight')#, vmin = -5, vmax = 5, cmap = 'PiYG')
plt.colorbar(a)

#%% phase image

i = 24

#dS image
im = np.angle(data[:, i, 1, :, :]) - np.angle(data[:, i, 0, :, :])
bg = np.angle(data[:, 0, 1, : :,]) - np.angle(data[:, 0, 0, :, :])
im = im - bg
phi = np.sum(im, axis=0)

plt.figure()
plt.imshow(np.sum(np.angle(data[:, i, 1, :, :]), axis = 0), cmap = 'twilight')

#%% propagate ROIs

size = 30
size2 = int(size/2)

# coordinates, propagated and unpropagated, image
ROIs = np.zeros(shape = (coordinates.shape[0], 2, size, size))
zmins = np.zeros(shape = coordinates.shape[0])

zrange = 1e-2
samples = 1500
iterations = 20

for i in range(coordinates.shape[0]):
    x,y = coordinates[i,:]
    image = data[:,0,0,:,:]
    image = np.sum(image, axis = 0)
    image = image[x-size2:x+size2, y-size2:y+size2]
    # def autofocus(image, numPix, spacePix, lam, zrange, samples, iterations):
    zmin, zs = cf.autofocus(image, size+1, 6.9e-06, 545e-09, zrange, samples, iterations)
    
    ROIs[i, 1, :, :] = cf.angularSpectrumMethod(image, size+1, 6.9e-6, 545e-9, zmin)
    ROIs[i, 0, :, :] = image
    zmins[i] = zmin
    
#%%
fig, (ax1, ax2) = plt.subplots(1,2)
for i in range(coordinates.shape[0]):

    vmin = np.min([ROIs[i, 1, :, :], ROIs[i, 0, :, :]])
    vmax = np.max([ROIs[i, 1, :, :], ROIs[i, 0, :, :]])
    
    ax1.imshow(ROIs[i, 0, :, :], origin = 'lower', cmap = 'gray', vmin = vmin, vmax = vmax)
    ax1.title.set_text('original')
    ax2.imshow(ROIs[i, 1, :, :], origin = 'lower', cmap = 'gray', vmin = vmin, vmax = vmax)
    ax2.title.set_text('propagated by' + '{:.2e}'.format(zmins[i]))
    plt.savefig(str(i)+'.png')
    # fig.clear(True)
    
#%% plot image
plt.figure()
plt.imshow(np.abs(data[1,0,0,:,:]), origin = 'lower')

#%% plot pump probe of each pixel at index 22

ran = 50 # range to evaluate pump probe
cent = 22

pp = np.abs(data[:, 10:ran, 1, :, :]) - np.abs(data[:, 10:ran, 0, :, :])
ppstd = np.std(pp, axis = (0, 2, 3))

plt.figure(11)
plt.plot(times[10:ran], pp)

#take std dev.