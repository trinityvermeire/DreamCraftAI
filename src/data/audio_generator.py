import math
import numpy as np
import pyaudio
import threading

import random

SAMPLE_RATE = 44100
TIME_PER_FREQUENCY = 5 # In seconds
BITDEPTH = 16
OFFSET = 6

class AudioGenerator:
    def __init__(self, seconds=TIME_PER_FREQUENCY):
        self.seconds = seconds

        # Initialize PyAudio
        self.pyaudio_instance = pyaudio.PyAudio()

        # Queue to hold frequencies
        self.frequency_queue = []

        # Flag to control the playback loop
        self.playback_flag = False

        # Open a stream
        self.stream = self.pyaudio_instance.open(
            format=self.pyaudio_instance.get_format_from_width(2),
            channels=2,
            rate=SAMPLE_RATE,
            output=True
        )

    def generateBinauralAudioData(self, startFrequency, endFrequency):
        audioDataLeft = []
        audioDataRight = []
        numSamples = math.ceil(SAMPLE_RATE * self.seconds)  # Calculate the number of samples per audio cycle
        differencePerSample = (endFrequency - startFrequency) / numSamples  # Calculate the frequency difference between each sample assuming the transition is linear
        timePerSample = 1 / SAMPLE_RATE  # Calculate the time in seconds elapsed during each sample
        difference = startFrequency-endFrequency

        # Initializing variables to lower overhead during for loop
        time = 0
        frequency = startFrequency

        smoothing_factor = 1/self.seconds

        # Generate audio samples for the given duration
        for i in range(numSamples):
            time += timePerSample  # Calculate time in seconds for the current sample
            frequency += differencePerSample  # Calculate frequency for the current sample

            position = np.pi*time*2 # Simply for computing speed so this doensn't get computed 3 times

            #offset = OFFSET*np.cos(position*smoothing_factor)
            individualOffset = OFFSET / 2  # Calculate offset for each ear

            smoother = difference*(np.cos(position/4 * smoothing_factor)+1)/2+endFrequency

            # Calculate amplitude of the audio sample (sine wave) for left and right ears
            amplitudeLeft = np.sin(position * (smoother - individualOffset))  # Apply phase shift for left ear
            amplitudeRight = np.sin(position * (smoother + individualOffset))  # Apply phase shift for right ear

            scaling_factor = 2 ** (BITDEPTH - 1)

            # Convert amplitude to the range of the bit depth (e.g., 16-bit)
            # Scale the amplitude values to the range of the chosen bit depth
            sampleValueLeft = int(amplitudeLeft * scaling_factor)
            sampleValueRight = int(amplitudeRight * scaling_factor)

            # Add the sample values to the audio data lists for left and right ears
            audioDataLeft.append(sampleValueLeft)
            audioDataRight.append(sampleValueRight)

        # Convert to bytes
        audioDataLeft = np.array(audioDataLeft)
        audioDataRight = np.array(audioDataRight)
        audio_bytes_stereo = np.column_stack((audioDataLeft, audioDataRight)).astype(np.int16).tobytes()


        self.current_audio = audio_bytes_stereo

    def play_audio(self, audio_bytes_stereo):
        if not audio_bytes_stereo:
            return
        self.stream.write(audio_bytes_stereo)


    def playback_loop(self):
        self.current_audio = False
        while self.playback_flag:
            startFrequency = self.frequency_queue[0]
            if len(self.frequency_queue)!=1: self.frequency_queue.pop(0)
            endFrequency = self.frequency_queue[0]
            thread = threading.Thread(target=self.generateBinauralAudioData, args=(startFrequency, endFrequency))
            thread.start()  # Start the thread to generate the audio
            self.play_audio(self.current_audio)
            thread.join() # Waits for the thread to be finished

    def play(self):
        if not self.playback_flag:
            self.playback_flag = True
            self.stream.start_stream()
            threading.Thread(target=self.playback_loop).start()

    def pause(self):
        self.playback_flag = False
        self.stream.stop_stream()

    def close(self):
        self.frequency_queue.clear()
        self.stream.close()

    def clear_queue(self):
        self.frequency_queue.clear()

    def add_frequency(self, frequency):
        if frequency > OFFSET/2:
            self.frequency_queue.append(frequency)
            print(f'{frequency}hz added to the queue')

'''
# Example usage:
player = AudioGenerator()

# Add frequencies with offsets to the queue
lowest_range = max(OFFSET/2+1,30)   # Lowest frequency that can be added to the queue
highest_range = 90                  # Highest frequency that can be added to the queue
variation = 20                      # Amount that the frequency can change every time interval
current = 50                        # Starting frequency
for _ in range(20):
    current += random.randint(max(lowest_range-current, -variation), min(highest_range-current, variation))
    print(f"{current}hz")
    player.add_frequency(current)


# Starts the audio
player.play()
'''