import pygame
import time
import os

# Path to the directory with the audio clips
AUDIO_DIR = os.path.dirname(os.path.abspath(__file__)) + "/audio_clips/"

actions = ['0.9_Hz_Delta',
           '3_Hz_Delta',
           '4_Hz_Delta_Isochronic_Pulses',
           '6_Hz_Theta_Isochronic_Pulses',
           '7_Hz_Theta',
           '10_Hz__Alpha_Isochronic_Pulses',
           '12_Hz_Alpha',
           '20_Hz_Beta',
           '40_Hz_Gamma',
           '40_Hz_Gamma_Isochronic_Pulses'
           ]

class AudioPlayer:
    def __init__(self):
        pygame.mixer.init()
        self.channel1 = pygame.mixer.Channel(0)
        self.channel2 = pygame.mixer.Channel(1)
        self.current_channel = self.channel1  # Start with channel 1
        self.queue = []
        self.current_audio = None
        self.paused = False

    def play(self):
        self.play_next()

    def play_next(self):
        if len(self.queue) != 0:
            next_audio = self.queue[0]
            if len(self.queue) != 1: self.queue.pop(0)
            print(next_audio)
            self.current_audio = next_audio
            sound = pygame.mixer.Sound(os.path.join(AUDIO_DIR, next_audio))
            length = sound.get_length()
            self.current_channel.play(sound, fade_ms=int(length*500))  # Fade in the new audio on the new channel
            self.current_channel = self.channel2 if self.current_channel == self.channel1 else self.channel1  # Alternate channels
            self.check_queue(sound.get_length())
            
    def check_queue(self, duration):
        time.sleep(duration*0.5)
        pygame.mixer.fadeout(int(duration*500))
        self.play_next()



    def pause(self):
        pygame.mixer.music.pause()
        self.paused = True

    def resume(self):
        pygame.mixer.music.unpause()
        self.paused = False

    def add_queue(self, filename):
        self.queue.append(filename)

    def clear_queue(self):
        self.queue.clear()

'''
if __name__ == "__main__":
    player = AudioPlayer()
    
    for _ in range(500):
        for action in actions[1:]:
            player.add_queue(action+".mp3")
        for action in actions[-2::-1]:
            player.add_queue(action+".mp3")

    
    player.play()  # Start playing the audio files in the queue
'''
