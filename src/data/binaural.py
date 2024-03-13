#!/usr/bin/env python
# encoding: utf-8
 
"""
from: http://en.wikipedia.org/wiki/Binaural_beats
 
Binaural beats or binaural tones are auditory processing artifacts, or apparent sounds, 
the perception of which arises in the brain for specific physical stimuli. This effect 
was discovered in 1839 by Heinrich Wilhelm Dove.
 
The brain produces a phenomenon resulting in low-frequency pulsations in the loudness 
and sound localization of a perceived sound when two tones at slightly different frequencies 
are presented separately, one to each of a subject's ears, using stereo headphones. A 
beating tone will be perceived, as if the two tones mixed naturally, out of the brain. 
The frequency of the tones must be below about 1,000 to 1,500 hertz for the beating to 
be heard. The difference between the two frequencies must be small (below about 30 Hz) 
for the effect to occur; otherwise, the two tones will be heard separately and no beat 
will be perceived.
 
Binaural beats are of interest to neurophysiologists investigating the sense of hearing. 
Second, binaural beats reportedly influence the brain in more subtle ways through the 
entrainment of brainwaves[1][2] and can be used to reduce anxiety[3] and provide other 
health benefits such as control over pain.[4]
 
props to:
1) http://mail.python.org/pipermail/python-list/2009-June/1207339.html
2) http://www.daniweb.com/code/snippet263775.html
"""
 
import math
import wave
import struct
import array
 
def make_soundfile(left_freq=440, right_freq=460, data_size=10000, fname="test.wav"):
    """
    create a synthetic 'sine wave' wave file with frequency freq
    file fname has a length of about data_size * 2
    """
    frate = 11025.0  # framerate as a float
    amp = 8000.0     # multiplier for amplitude
 
    # make a sine list ...
    sine_list = []
    for x in range(data_size):
        left = math.sin(2*math.pi*left_freq*(x/frate))
        right = math.sin(2*math.pi*right_freq*(x/frate))
        sine_list.append((left,right))
 
    # get ready for the wave file to be saved ...
    wav_file = wave.open(fname, "w")
    # give required parameters
    nchannels = 2
    sampwidth = 2
    framerate = int(frate)
    nframes = data_size
    comptype = "NONE"
    compname = "not compressed"
    # set all the parameters at once
    wav_file.setparams((nchannels, sampwidth, framerate, nframes,
        comptype, compname))
    # now write out the file ...
    print( "may take a moment ..." )
    for s in sine_list:
        data = array.array('h')
        data.append(int(s[0]*amp/2)) # left channel
        data.append(int(s[1]*amp/2)) # write channel
        # write the audio frames to file
        wav_file.writeframes(data)
    wav_file.close()
    print( "%s written" % fname )
 
 
# set some variables ...
left_freq = 4.0
right_freq = 8.0
# data size, file size will be about 2 times that
# duration is about 4 seconds for a data_size of 40000
data_size = 60000
 
# write the synthetic wave file to ...
fname = "binaural_%s_%s.wav" % (left_freq, right_freq)
 
make_soundfile(left_freq, right_freq, data_size, fname)
