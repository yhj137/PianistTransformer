from src.utils.midi import ids_to_midi, midi_to_ids
from src.model.pianoformer import PianoT5Gemma
from torch.nn.utils.rnn import pad_sequence
from transformers import LogitsProcessorList, LogitsProcessor
from tqdm import tqdm
import torch
from src.utils.midi import normalize_midi, merge_and_sort
from miditoolkit import MidiFile, Note, TempoChange, Instrument, ControlChange
import bisect

class BatchSparseForcedTokenProcessor(LogitsProcessor):
    def __init__(self, input_ids, config, target_len, origin_len, already, weight, progress_callback):
        self.batch_map = [{j: input_ids[i][j] for j in range(0, len(input_ids[i]), 8)} for i in range(len(input_ids))]
        self.valid_id_range = config.valid_id_range
        self.target_len = target_len
        self.origin_len = origin_len
        self.already = already
        self.weight = weight
        self.progress_callback = progress_callback

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.progress_callback:
            self.progress_callback(
                (input_ids.shape[1]) / (self.target_len) * self.weight + self.already
            )
        step = input_ids.shape[1] - 1
        batch_size = scores.shape[0]
        for i in range(batch_size):
            sample_map = self.batch_map[i]
            if step in sample_map:
                forced_token_id = sample_map[step]
                scores[i] = float('-inf')
                scores[i, forced_token_id] = 0.0
            else:
                step = step % 8
                scores[i, :self.valid_id_range[step][0]] = float('-inf')
                scores[i, self.valid_id_range[step][1]:] = float('-inf')
                #if step % 8 > 3:
                #    scores = scores / 0.95
        return scores

@torch.no_grad()
def batch_performance_render(
        model, 
        score_midi_objs, 
        max_context_length=4096, 
        overlap_ratio=0.5, 
        temperature=1.0,
        top_p=0.95,
        device="cpu",
        progress_callback=None
    ):
    def slide_window(total_len, window_len):
        if total_len <= window_len:
            return [(0, total_len)]
        window_len = window_len // 8 * 8
        out = []
        start = 0
        while start + window_len <= total_len:
            out.append((start, start + window_len))
            start += int(window_len * (1 - overlap_ratio)) // 8 * 8
        if out[-1][1] != total_len:
            out.append((start, total_len))
        return out
    if max_context_length > 4096:
        raise ValueError("You should set max_context_length <= 4096!")
    ids_list = [midi_to_ids(model.config, score_midi_obj) for score_midi_obj in score_midi_objs]
    batch_ids = [torch.tensor(midi_to_ids(model.config, score_midi_obj), dtype=torch.long).to(device) for score_midi_obj in score_midi_objs]
    len_list = [len(batch_ids[i]) for i in range(len(batch_ids))]
    
    input_ids = pad_sequence(batch_ids, batch_first=True, padding_value=model.config.pad_token_id)
    windows = slide_window(input_ids.shape[1], max_context_length)
    #print(windows)
    output_list = []
    res_tensor = None
    for i in tqdm(range(len(windows))):
        start, end = windows[i]
        logits_processor = LogitsProcessorList([
            BatchSparseForcedTokenProcessor(
                input_ids[:,start:end], 
                model.config, 
                end - start,
                start,
                i / len(windows),
                1 / len(windows),
                progress_callback,
            )
        ])
        if i == 0:
            output = model.generate(
                input_ids[:,start:end], 
                do_sample=True, 
                max_new_tokens=end-start, 
                logits_processor=logits_processor,
                temperature=temperature,
                top_p=top_p,
            )
            res_tensor = output[:,1:]
        else:
            last_start, last_end = windows[i-1]
            length = int(((last_end-last_start) - (start-last_start)) * 0.2)
            decoder_input_ids = output_list[i-1][:, start-last_start:last_end-last_start - length]
            start_tensor = torch.tensor([[model.config.bos_token_id] for _ in range(input_ids.shape[0])], dtype=torch.long).to(device)
            decoder_input_ids = torch.cat([start_tensor, decoder_input_ids], dim=1)
            #print(decoder_input_ids.shape)
            output = model.generate(
                input_ids[:,start:end], 
                decoder_input_ids=decoder_input_ids,
                do_sample=True, 
                max_new_tokens=end-last_end+length, 
                logits_processor=logits_processor,
                temperature=temperature,
                top_p=top_p,
            )
            res_tensor = torch.cat([res_tensor[:,:-length], output[:,-(end-last_end+length):]], dim=1)
        output_list.append(output)
    res_tensor = res_tensor.cpu().numpy().tolist()
    #print(res_tensor)
    res = []
    for i in range(len(res_tensor)):
        #print(res_tensor[i][:len_list[i]])
        res.append(ids_to_midi(model.config, res_tensor[i][:len_list[i]], ref=ids_list[i]))
    return res


def map_midi(score_midi_obj, performance_midi_obj):
    def compute_duration(start_time, target_duration, tempo_list):
        if target_duration <= 0:
            return 0
        if not tempo_list:
            tempo_list = [TempoChange(120, 0)]
        tempo_times = [t.time for t in tempo_list]
        start_tempo_idx = bisect.bisect_right(tempo_times, start_time) - 1
        if start_tempo_idx < 0:
            start_tempo_idx = 0
        total_ticks_duration = 0.0
        time_remaining_ms = float(target_duration)
        current_tick = start_time
        current_tempo_idx = start_tempo_idx
        while time_remaining_ms > 1e-9:
            current_tempo_event = tempo_list[current_tempo_idx]
            current_bpm = current_tempo_event.tempo
            ms_per_tick = (60 * 1000.0 / current_bpm) / 500
            end_of_segment_tick = float('inf')
            if current_tempo_idx + 1 < len(tempo_list):
                end_of_segment_tick = tempo_list[current_tempo_idx + 1].time
            ticks_in_segment = end_of_segment_tick - current_tick
            ms_in_segment = ticks_in_segment * ms_per_tick
            if time_remaining_ms <= ms_in_segment:
                ticks_needed = time_remaining_ms / ms_per_tick
                total_ticks_duration += ticks_needed
                time_remaining_ms = 0
            else:
                total_ticks_duration += ticks_in_segment
                time_remaining_ms -= ms_in_segment
                current_tick = end_of_segment_tick
                current_tempo_idx += 1
        return round(total_ticks_duration)
    
    norm_score = merge_and_sort(score_midi_obj)
    norm_performance= normalize_midi(performance_midi_obj)
    score_tempo_change_points = []
    last_time = -1
    assert(len(norm_score.instruments[0].notes) == len(norm_performance.instruments[0].notes))
    for i, note in enumerate(norm_score.instruments[0].notes):
        if abs(note.start - last_time) > 20:
            last_time = note.start
            p_start = norm_performance.instruments[0].notes[i].start
            score_tempo_change_points.append(
                (
                    note.start,
                    p_start,
                    i
                )
            )

    tempo_changes = []
    tempo_changes_list = []
    for i in range(len(score_tempo_change_points) - 1):
        s_ioi = score_tempo_change_points[i+1][0] - score_tempo_change_points[i][0]
        p_ioi = score_tempo_change_points[i+1][1] - score_tempo_change_points[i][1]
        tempo = max(min(round(s_ioi / p_ioi * 120), 300), 10)
        tempo_changes.append(TempoChange(tempo, score_tempo_change_points[i][0]))
        tempo_changes_list.append((tempo, score_tempo_change_points[i][2]))
    norm_score.tempo_changes = tempo_changes

    notes = norm_score.instruments[0].notes
    performance_notes = norm_performance.instruments[0].notes
    for i in range(len(tempo_changes_list) - 1):
        start_id = tempo_changes_list[i][1]
        p_length = performance_notes[tempo_changes_list[i+1][1]].start - performance_notes[tempo_changes_list[i][1]].start
        s_length = notes[tempo_changes_list[i+1][1]].start - notes[tempo_changes_list[i][1]].start
        for j in range(tempo_changes_list[i][1] + 1, tempo_changes_list[i+1][1]):
            p_ioi = performance_notes[j].start - performance_notes[start_id].start
            notes[j].start = notes[start_id].start + int((p_ioi / p_length) * s_length)

    for i in range(len(notes)):
        notes[i].velocity = performance_notes[i].velocity
        notes[i].end = notes[i].start + compute_duration(notes[i].start, performance_notes[i].end - performance_notes[i].start, tempo_changes)

    score_tempo_change_points.insert(
        0,
        (
            0,
            0,
            -1
        )
    )
        
    score_tempo_change_points.append(
        (
            score_tempo_change_points[-1][0] + 5000,
            score_tempo_change_points[-1][1] + 5000,
            len(score_tempo_change_points)
        )
    )

    new_cc_list = []
    cc_list = norm_performance.instruments[0].control_changes
    cnt = 0
    for i in range(len(cc_list)):
        while not (score_tempo_change_points[cnt][1] <= cc_list[i].time < score_tempo_change_points[cnt+1][1]):
            cnt += 1
        ratio = (cc_list[i].time - score_tempo_change_points[cnt][1]) / (score_tempo_change_points[cnt+1][1] - score_tempo_change_points[cnt][1])
        new_cc_list.append(
            ControlChange(
                64,
                cc_list[i].value,
                score_tempo_change_points[cnt][0] + round(ratio * (score_tempo_change_points[cnt+1][0] - score_tempo_change_points[cnt][0]))
            )
        )
    norm_score.instruments[0].control_changes = new_cc_list
    return norm_score

if __name__ == "__main__":
    score = MidiFile("temp/s.mid")
    performance = MidiFile("temp/rebuild.mid")
    aligned = map_midi(score, performance)
    aligned.dump("temp/a.mid")