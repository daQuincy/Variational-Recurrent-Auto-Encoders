# https://github.com/craffel/pretty-midi/blob/master/examples/reverse_pianoroll.py

import numpy as np
import pretty_midi

class PianoRollToMIDI:
    @staticmethod
    def convert(piano_roll, fs=100, base_note=0, program=0, resolution=480, initial_tempo=120):
        notes, frames = piano_roll.shape
        pm = pretty_midi.PrettyMIDI(resolution=resolution, initial_tempo=initial_tempo)
        instrument = pretty_midi.Instrument(program=program)
        
        # pad 1 column of zeros so we can acknowledge inital and ending events
        piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], "constant")
        
        # use changes in velocities to find note on / note off events                       
        velocity_changes = np.nonzero(np.diff(piano_roll).T)
        
        # keep track on velocities and note on times
        prev_velocities = np.zeros(notes, dtype=int)
        note_on_time = np.zeros(notes)
        
        for time, note in zip(*velocity_changes):
            # time +1 cause of padding
            velocity = piano_roll[note, time+1]
            time = time / fs
            if velocity > 0:
                if prev_velocities[note] == 0:
                    note_on_time[note] = time
                    prev_velocities[note] = velocity
            else:
                pm_note = pretty_midi.Note(velocity=prev_velocities[note],
                                           pitch=note,
                                           start=note_on_time[note],
                                           end=time)
                instrument.notes.append(pm_note)
                prev_velocities[note] = 0
                
        pm.instruments.append(instrument)
        
        return pm
            