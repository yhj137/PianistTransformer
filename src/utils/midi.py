from miditoolkit import MidiFile, Note, Instrument, TempoChange, ControlChange
import bisect
import numpy as np
import os
from copy import copy
import random
from collections import defaultdict
import json

def normalize_midi(midi_obj, target_ticks_per_beat=500, target_tempo=120):
    output_midi_obj = MidiFile(ticks_per_beat=target_ticks_per_beat)
    output_midi_obj.tempo_changes.append(TempoChange(target_tempo, 0))
    
    tick_to_time_map = midi_obj.get_tick_to_time_mapping()
    seconds_to_target_ticks_factor = target_ticks_per_beat * (target_tempo / 60.0)

    all_converted_notes = []
    for instrument in midi_obj.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                start_time_sec = tick_to_time_map[note.start]
                end_time_sec = tick_to_time_map[note.end]
                
                new_start_tick = round(start_time_sec * seconds_to_target_ticks_factor)
                new_end_tick = round(end_time_sec * seconds_to_target_ticks_factor)
                
                if new_start_tick >= new_end_tick:
                    new_end_tick = new_start_tick + 1

                all_converted_notes.append(
                    Note(
                        velocity=note.velocity, 
                        pitch=note.pitch, 
                        start=new_start_tick, 
                        end=new_end_tick
                    )
                )

    notes_by_pitch = defaultdict(list)
    for note in all_converted_notes:
        notes_by_pitch[note.pitch].append(note)

    merged_notes = []
    for pitch in sorted(notes_by_pitch.keys()):
        sorted_notes = sorted(notes_by_pitch[pitch], key=lambda n: n.start)
        
        if len(sorted_notes) > 1:
            for i in range(len(sorted_notes) - 1):
                current_note = sorted_notes[i]
                next_note = sorted_notes[i+1]
                
                if current_note.end >= next_note.start:
                    current_note.end = next_note.start
                    if current_note.start >= current_note.end:
                        current_note.pitch = -1
                        
        merged_notes.extend([n for n in sorted_notes if n.pitch != -1])

    merged_cc = []
    for instrument in midi_obj.instruments:
        if not instrument.is_drum:
            for cc in instrument.control_changes:
                if cc.number == 64:
                    time_sec = tick_to_time_map[cc.time]
                    new_time_tick = round(time_sec * seconds_to_target_ticks_factor)
                    merged_cc.append(
                        ControlChange(
                            number=64, 
                            value=cc.value, 
                            time=new_time_tick
                        )
                    )

    merged_notes.sort(key=lambda x: (x.start, x.pitch))
    merged_cc.sort(key=lambda x: (x.time, x.number))
    
    output_instrument = Instrument(program=0, is_drum=False, name="Piano")
    output_instrument.notes = merged_notes
    output_instrument.control_changes = merged_cc
    output_midi_obj.instruments.append(output_instrument)
    
    max_tick = 0
    if output_instrument.notes:
        max_tick = max(max_tick, max(n.end for n in output_instrument.notes if n.end is not None))
    if output_instrument.control_changes:
        max_tick = max(max_tick, max(c.time for c in output_instrument.control_changes if c.time is not None))
    
    output_midi_obj.max_tick = max_tick + target_ticks_per_beat 
    return output_midi_obj

def merge_and_sort(midi_obj):
    output_midi_obj = MidiFile(ticks_per_beat=500)
    output_midi_obj.time_signature_changes = midi_obj.time_signature_changes
    output_midi_obj.key_signature_changes = midi_obj.key_signature_changes

    output_instrument = Instrument(program=0, is_drum=False, name="Piano")
    tick_ratio = 500 / midi_obj.ticks_per_beat
    all_notes = []
    for instrument in midi_obj.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                all_notes.append(
                    Note(
                        velocity=note.velocity, 
                        pitch=note.pitch, 
                        start=round(note.start * tick_ratio), 
                        end=round(note.end * tick_ratio)
                    )
                )
    notes_by_pitch = defaultdict(list)
    for note in all_notes:
        notes_by_pitch[note.pitch].append(note)
    merged_notes = []
    for pitch in sorted(notes_by_pitch.keys()):
        sorted_notes = sorted(notes_by_pitch[pitch], key=lambda n: n.start)
        if len(sorted_notes) > 1:
            for i in range(len(sorted_notes) - 1):
                current_note = sorted_notes[i]
                next_note = sorted_notes[i+1]
                if current_note.end >= next_note.start:
                    current_note.end = next_note.start
                    if current_note.start >= current_note.end:
                        current_note.pitch = -1

        merged_notes.extend([n for n in sorted_notes if n.pitch != -1])

    merged_notes.sort(key=lambda x: (x.start, x.pitch))
    output_instrument.notes = merged_notes
    output_midi_obj.instruments.append(output_instrument)
    for time_signature in output_midi_obj.time_signature_changes:
        time_signature.time = round(time_signature.time * tick_ratio)
    for key_signature in output_midi_obj.key_signature_changes:
        key_signature.time = round(key_signature.time * tick_ratio)
    return output_midi_obj

def midi_to_ids(config, midi_obj, normalize=True):
    def get_pedal(time_list, ccs, time):
        i = bisect.bisect_right(time_list, time)
        if i == 0:
            return 0
        else:
            return ccs[i-1].value
    if normalize:
        norm_midi_obj = normalize_midi(midi_obj)
    else:
        norm_midi_obj = midi_obj
    time_list = [cc.time for cc in norm_midi_obj.instruments[0].control_changes]
    #print(time_list)
    intervals = []
    last_time = 0
    for note in norm_midi_obj.instruments[0].notes:
        intervals.append(note.start - last_time)
        last_time = note.start
    intervals.append(4990)

    ids = []
    last_time = 0
    for i, note in enumerate(norm_midi_obj.instruments[0].notes):
        interval = config.timing_start + intervals[i]
        #print(interval - interval_start)

        pitch = config.pitch_start + note.pitch
        velocity = config.velocity_start + note.velocity
        duration = config.timing_start + note.duration
        last_time = last_time + intervals[i]

        pedal1 = config.pedal_start + get_pedal(time_list, norm_midi_obj.instruments[0].control_changes, last_time)
        pedal2 = config.pedal_start + get_pedal(time_list, norm_midi_obj.instruments[0].control_changes, last_time + intervals[i+1] * 1 / 4)
        pedal3 = config.pedal_start + get_pedal(time_list, norm_midi_obj.instruments[0].control_changes, last_time + intervals[i+1] * 2 / 4)
        pedal4 = config.pedal_start + get_pedal(time_list, norm_midi_obj.instruments[0].control_changes, last_time + intervals[i+1] * 3 / 4)
        
        pitch = min(config.valid_id_range[0][1] - 1, max(config.valid_id_range[0][0], pitch))
        interval = min(config.valid_id_range[1][1] - 1, max(config.valid_id_range[1][0], interval))
        velocity = min(config.valid_id_range[2][1] - 1, max(config.valid_id_range[2][0], velocity))
        duration = min(config.valid_id_range[3][1] - 1, max(config.valid_id_range[3][0], duration))
        pedal1 = min(config.valid_id_range[4][1] - 1, max(config.valid_id_range[4][0], pedal1))
        pedal2 = min(config.valid_id_range[5][1] - 1, max(config.valid_id_range[5][0], pedal2))
        pedal3 = min(config.valid_id_range[6][1] - 1, max(config.valid_id_range[6][0], pedal3))
        pedal4 = min(config.valid_id_range[7][1] - 1, max(config.valid_id_range[7][0], pedal4))

        ids.extend([pitch, interval, velocity, duration, pedal1, pedal2, pedal3, pedal4])
    return ids

def ids_to_midi(config, ids, target_ticks_per_beat = 500, target_tempo = 120, ref = None):
    note_list = []
    cc_list = []
    intervals = []
    for i in range(0, len(ids), 8):
        if ref:
            if ref[i+1]  - config.timing_start > 0:
                intervals.append(max(5, ids[i+1] - config.timing_start))
            else:
                intervals.append(ids[i+1] - config.timing_start)

        else:
            intervals.append(ids[i+1] - config.timing_start)
    intervals.append(4990)
    
    last_time = 0
    for i in range(0, len(ids), 8):
        interval = intervals[i // 8]
        pitch = ids[i] - config.pitch_start
        velocity = ids[i+2] - config.velocity_start
        duration = ids[i+3] - config.timing_start
        pedal1 = ids[i+4] - config.pedal_start
        pedal2 = ids[i+5] - config.pedal_start
        pedal3 = ids[i+6] - config.pedal_start
        pedal4 = ids[i+7] - config.pedal_start
        note_list.append(Note(velocity, pitch, last_time + interval, last_time + interval + duration))
        last_time += interval
        
        interval_time = intervals[i // 8 + 1]
        interval_step = intervals[i // 8 + 1] / 4

        cc_list.append(ControlChange(64, pedal1, last_time))
        cc_list.append(ControlChange(64, pedal2, round(last_time + interval_step)))
        cc_list.append(ControlChange(64, pedal3, round(last_time + interval_time - interval_step * 2)))
        cc_list.append(ControlChange(64, pedal4, round(last_time + interval_time - interval_step)))

        #cc_list.append(ControlChange(64, pedal1, last_time))
        #cc_list.append(ControlChange(64, pedal2, round(last_time + intervals[i // 8 + 1] * 1 / 4)))
        #cc_list.append(ControlChange(64, pedal3, round(last_time + intervals[i // 8 + 1] * 2 / 4)))
        #cc_list.append(ControlChange(64, pedal4, round(last_time + intervals[i // 8 + 1] * 3 / 4)))

    last_value = 0
    new_cc_list = []
    for cc in cc_list:
        if cc.value != last_value:
            new_cc_list.append(cc)
        last_value = cc.value

    max_tick = 0
    for note in note_list:
        max_tick = max(max_tick, note.end)
    for cc in cc_list:
        max_tick = max(max_tick, cc.time)
    max_tick = max_tick + 1

    output = MidiFile(ticks_per_beat=target_ticks_per_beat)
    output.instruments.append(Instrument(program=0, is_drum=False, name="Piano", notes=note_list, control_changes=new_cc_list))
    output.tempo_changes.append(TempoChange(target_tempo, 0))
    output.max_tick = max_tick
    
    return output

def read_corresp(corresp_path):
    out = []
    performacne_id_list = []
    with open(corresp_path, "r") as f:
        align_txt = f.readlines()

    score_ids_map = {}
    performance_ids_map = {}
    score_temp_list = []
    performance_temp_list = set()
    for line in align_txt[1:]:
        informs = line.split("\t")
        if informs[0] != '*':
            score_temp_list.append((float(informs[1]), int(informs[3]), int(informs[0])))
        if informs[5] != '*':
            performance_temp_list.add((float(informs[6]), int(informs[8]), int(informs[5])))
    performance_temp_list = list(performance_temp_list)
    score_temp_list.sort()
    performance_temp_list.sort()
    for i, inform in enumerate(score_temp_list):
        score_ids_map[inform[2]] = i
    for i, inform in enumerate(performance_temp_list):
        performance_ids_map[inform[2]] = i

    for line in align_txt[1:]:
        informs = line.split("\t")
        if informs[0] == '*':
            break
        if informs[5] != '*':
            out.append((score_ids_map[int(informs[0])], performance_ids_map[int(informs[5])]))
        else:
            out.append((score_ids_map[int(informs[0])], -1))
    
    for line in align_txt[1:]:
        informs = line.split("\t")
        if informs[5] != '*':
            performacne_id_list.append(performance_ids_map[int(informs[5])])
    if out[0][1] == -1:
        out[0] = (out[0][0], min(performacne_id_list))
    if out[-1][1] == -1:
        out[-1] = (out[-1][0], max(performacne_id_list)) 
    out.sort()
    return out

def interpolate(a, b):
    a = np.array(a) + np.linspace(0, 1e-5, len(a))
    b = np.array(b)
    known_inds = np.where(~np.isnan(b))[0]
    x_known = a[known_inds]
    y_known = b[known_inds]
    res = np.interp(a, x_known, y_known)
    res[known_inds] = b[known_inds]
    return [round(i) for i in res.tolist()]

def segment_sequences(x, label, unknown_ids, total_notes, max_consecutive_missing, min_segment_notes):

    if not unknown_ids:
        if total_notes >= min_segment_notes:
            return [x], [label]
        else:
            return [], []

    x_segments = []
    label_segments = []
    
    unknown_set = set(unknown_ids)
    
    last_cut_note_idx = 0
    consecutive_missing_count = 0

    for i in range(total_notes):
        if i in unknown_set:
            consecutive_missing_count += 1
        else:
            consecutive_missing_count = 0

        if consecutive_missing_count >= max_consecutive_missing:
            segment_end_note_idx = i - consecutive_missing_count + 1
            
            if segment_end_note_idx - last_cut_note_idx >= min_segment_notes:
                start_token = last_cut_note_idx * 8
                end_token = segment_end_note_idx * 8
                
                x_segments.append(x[start_token:end_token])
                label_segments.append(label[start_token:end_token])
            
            last_cut_note_idx = i + 1
            consecutive_missing_count = 0

    if total_notes - last_cut_note_idx >= min_segment_notes:
        start_token = last_cut_note_idx * 8
        x_segments.append(x[start_token:])
        label_segments.append(label[start_token:])
        
    return x_segments, label_segments

def align_score_and_performance(config, score_midi_obj, performance_midi_obj):
    norm_score_midi_obj = normalize_midi(score_midi_obj)
    norm_performance_midi_obj = normalize_midi(performance_midi_obj)
    
    norm_score_midi_obj.dump("temp/score.mid")
    norm_performance_midi_obj.dump("temp/performance.mid")

    os.chdir("./tools/AlignmentTool")
    os.system(f"timeout 120s ./MIDIToMIDIAlign.sh ../../temp/performance ../../temp/score")
    os.chdir("./../../") 

    corresp_list = read_corresp("temp/score_corresp.txt")
    aligned_midi_obj = MidiFile(ticks_per_beat=500)
    score_notes = norm_score_midi_obj.instruments[0].notes
    performance_notes = norm_performance_midi_obj.instruments[0].notes
    score_start_list = []
    output_notes = []
    output_ccs = []
    vel_list = []
    start_list = []
    duration_list = []
    unknown_ids = []
    for i, ids in enumerate(corresp_list):
        if ids[1] != -1:
            vel_list.append(performance_notes[ids[1]].velocity)
            start_list.append(performance_notes[ids[1]].start)
            duration_list.append(performance_notes[ids[1]].end - performance_notes[ids[1]].start)
        else:
            vel_list.append(np.nan)
            duration_list.append(np.nan)
            unknown_ids.append(i)
        score_start_list.append(score_notes[ids[0]].start)
    start_list.sort()
    temp = []
    cnt = 0
    for i in range(len(corresp_list)):
        if i not in unknown_ids:
            temp.append(start_list[cnt])
            cnt += 1
        else:
            temp.append(np.nan)
    start_list = interpolate(score_start_list, temp)
    vel_list = interpolate(start_list, vel_list)
    duration_list = interpolate(start_list, duration_list)

    end_list = []
    for i, ids in enumerate(corresp_list):
        end = start_list[i]+duration_list[i]
        end_list.append(end)
        output_notes.append(Note(vel_list[i], score_notes[ids[0]].pitch, start_list[i], end))
    max_tick = max(end_list) + 4999
    for cc in norm_performance_midi_obj.instruments[0].control_changes:
        if cc.time <= max_tick:
            output_ccs.append(cc)
        else:
            break

    aligned_midi_obj.instruments.append(Instrument(program=0, is_drum=False, name="Piano", notes=output_notes, control_changes=output_ccs))
    x = midi_to_ids(config, norm_score_midi_obj)
    label = midi_to_ids(config, aligned_midi_obj, normalize=False)
    assert(len(x) == len(label))
    for i in range(len(x)):
        if i % 8 == 0:
            assert(x[i] == label[i])

    total_notes = len(score_notes)
    xs, labels = segment_sequences(
        x,
        label,
        unknown_ids,
        total_notes,
        5,
        64,
    )
    return xs, labels

def enhanced_ids(config, ids):
    res = copy(ids)
    retry = 10
    for i in range(len(res)):
        j = i % 8
        if j == 3:
            value = res[i] - config.valid_id_range[j][0]
            if value == 10:
                noise = 0
                for _ in range(retry):
                    n = round(np.random.randn() * 5)
                    if n >= -9 and n <= 5:
                        noise = n
                        break
            else:
                noise = 0
                for _ in range(retry):
                    n = round(np.random.randn() * 5)
                    if n >= -4 and n <= 5:
                        noise = n
                        break
            value = min(max(value + noise, 0), 4999)
            res[i] = config.valid_id_range[j][0] + value
        elif j == 2:
            value = res[i] - config.valid_id_range[j][0]
            if value == 5:
                noise = 0
                for _ in range(retry):
                    n = round(np.random.randn() * 2.5)
                    if n >= -4 and n <= 2:
                        noise = n
                        break
            elif value == 120:
                noise = 0
                for _ in range(retry):
                    n = round(np.random.randn() * 2.5)
                    if n >= -2 and n <= 7:
                        noise = n
                        break
            else:
                noise = 0
                for _ in range(retry):
                    n = round(np.random.randn() * 2.5)
                    if n >= -2 and n <= 2:
                        noise = n
                        break
            value = min(max(value + noise, 0), 127)
            res[i] = config.valid_id_range[j][0] + value
        elif j == 1:
            value = res[i] - config.valid_id_range[j][0]
            noise = 0
            for _ in range(retry):
                n = round(np.random.randn() * 5)
                if n >= -4 and n <= 5:
                    noise = n
                    break
            value = min(max(value + noise, 0), 4990)
            res[i] = config.valid_id_range[j][0] + value
    return res

def enhanced_ids_uniform(config, ids):
    res = copy(ids)
    for i in range(len(res)):
        j = i % 8
        if j == 3:
            value = res[i] - config.valid_id_range[j][0]
            if value == 10:
                noise = random.randint(-9, 5)
            else:
                noise = random.randint(-4, 5)
            value = min(max(value + noise, 0), 4999)
            res[i] = config.valid_id_range[j][0] + value
        elif j == 2:
            value = res[i] - config.valid_id_range[j][0]
            if value == 5:
                noise = random.randint(-4, 2)
            elif value == 120:
                noise = random.randint(-2, 7)
            else:
                noise = random.randint(-2, 2)
            value = min(max(value + noise, 0), 127)
            res[i] = config.valid_id_range[j][0] + value
        elif j == 1:
            value = res[i] - config.valid_id_range[j][0]
            noise = random.randint(-4, 5)
            value = min(max(value + noise, 0), 4990)
            res[i] = config.valid_id_range[j][0] + value
    return res

if __name__ == "__main__":
    from src.model.pianoformer import PianoT5GemmaConfig
    config = PianoT5GemmaConfig()
    with open("temp/out.json", "r") as f:
        ids = json.load(f)
    score_midi = MidiFile("temp/s.mid")
    score_ids = midi_to_ids(config, score_midi)
    midi = ids_to_midi(config, ids, ref=score_ids)
    midi.dump("temp/rebuild.mid")
