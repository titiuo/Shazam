import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from scipy.signal import iirfilter, freqz
import soundfile as sf
import IPython.display 
from scipy.signal import welch

son, Fe = sf.read('./wav/do_majeur.wav')
son = son[:,0]  # On ne garde qu'un seul canal audio

notes = {16.351 : 'Do', 17.323 : 'Do#', 18.354 : 'Ré', 19.445 : 'Re#', 20.601 : 'Mi', 21.827 : 'Fa',
          23.124 : 'Fa#', 24.499 : 'Sol', 25.956 : 'Sol#', 27.5 : 'La', 29.135 : 'La#', 30.868 : 'Si'} 

""" notes = {82: 'E', 110: 'A', 147: 'D', 196: 'G', 247: 'B'} """

found = []

def reponse_en_frequence(b, a):
    w,h = sig.freqz(b,a,4096)
    hdb = 20*np.log10(np.abs(h))
    plt.plot(w/2/np.pi,hdb)
    plt.grid()
    plt.xlabel(r'fréquence réduite $\nu$')
    plt.ylabel('dB')
    plt.title('Réponse en fréquence du filtre')
    plt.show()

def get_note(son, notes = notes):
    for note in notes.keys():
        #while note_mult < (Fe/2):
        for k in range(12):
            note_mult = note * 2**k
            print(f'Nous travaillons sur la note {notes[note]}...\n La frequence est {note_mult}')
            f_resonance = note_mult # Fréquence de résonance en Hz
            bandwidth = 1.9  # Bande passante en Hz
            fs = Fe  # Fréquence d'échantillonnage en Hz

            # Conception du filtre
            f_lower = f_resonance - bandwidth / 2  # Fréquence de coupure inférieure
            f_upper = f_resonance + bandwidth / 2  # Fréquence de coupure supérieure
            f_lower_normalized = f_lower*2 / fs  # Fréquence de coupure inférieure normalisée
            f_upper_normalized = f_upper*2 / fs  # Fréquence de coupure supérieure normalisée
            try:
                b, a = iirfilter(2, [f_lower_normalized, f_upper_normalized], btype='band', ftype='butter')
            except:
                print(f'Erreur pour la note {notes[note]}... \n à la fréquence {note_mult} Hz: f_lower_normalized = {f_lower_normalized} et f_upper_normalized = {f_upper_normalized}')
                continue
            #reponse_en_frequence(b, a)
            #son = son*np.hanning(len(son))  # Application d'une fenêtre de Hanning
            son_filtre = sig.lfilter(b, a, son)

            #IPython.display.Audio(son_filtre, rate=Fe)

            # Application du filtre et tracé du spectre
            freq, Pxx = welch(son, fs=fs)
            plt.semilogy(freq, Pxx, label='Signal d\'origine')
            #print(Pxx)
            seuil = 0.4*np.mean(Pxx)

            freq, Pxx_filtre = welch(son_filtre, fs=fs)
            plt.semilogy(freq, Pxx_filtre, label='Signal filtré')


            # Trouver l'indice correspondant à f_resonance 
            idx = np.abs(freq - f_resonance).argmin()

            # Vérifier si la valeur de Pxx_filtre à cet indice est supérieure au seuil
            print(f"La valeur de Pxx_filtre à {f_resonance} est {Pxx_filtre[idx]}")
            if Pxx_filtre[idx] > seuil:
                print(f"La valeur de Pxx_filtre à {f_resonance} est supérieure au seuil.")
                found.append(notes[note])
                break
    """             plt.title('Spectre du signal audio')
                plt.xlabel('Fréquence [Hz]')
                plt.ylabel('Densité spectrale de puissance')
                plt.legend()
                plt.grid(True)
                plt.plot([0, fs / 2], [seuil, seuil], 'r--' )
                plt.show(block = False)
                plt.ginput(1, timeout=-1)
                plt.close('all')  """
                

                

                    
    """         plt.title('Spectre du signal audio')
            plt.xlabel('Fréquence [Hz]')
            plt.ylabel('Densité spectrale de puissance')
            plt.legend()
            plt.grid(True)
            plt.plot([0, fs / 2], [seuil, seuil], 'r--' )
            plt.show(block = False)
            plt.ginput(1, timeout=-1)
            plt.close('all')  """
    return found
