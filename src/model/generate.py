from src.utils.midi import ids_to_midi, midi_to_ids
from src.model.pianoformer import PianoT5Gemma
from miditoolkit import MidiFile
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
                (input_ids.shape[1] - self.origin_len) / (self.target_len - self.origin_len) * self.weight + self.already
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
                end,
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
        res.append(ids_to_midi(model.config, res_tensor[i][:len_list[i]]))
    return res


def map_midi(score_midi_obj, performance_midi_obj):
    def compute_duration(start_time, target_duration, tempo_list):
        if target_duration <= 0:
            return 0
        if not tempo_list:
            # 如果没有提供tempo信息，则假定为默认的120 BPM
            tempo_list = [TempoChange(120, 0)]

        # --- 步骤1: 定位start_time所在的BPM区间 ---
        # 提取所有tempo变化的时间点
        tempo_times = [t.time for t in tempo_list]
        # 使用二分查找找到start_time应该插入的位置
        # bisect_right返回的是插入点索引，因此当前生效的tempo在索引-1的位置
        start_tempo_idx = bisect.bisect_right(tempo_times, start_time) - 1
        # 如果start_time在第一个tempo变化之前，索引会是-1，修正为0
        if start_tempo_idx < 0:
            start_tempo_idx = 0

        # --- 步骤2: 初始化循环变量 ---
        total_ticks_duration = 0.0
        time_remaining_ms = float(target_duration)
        current_tick = start_time
        current_tempo_idx = start_tempo_idx

        # --- 步骤3: 循环处理每个BPM区间，直到消耗完target_duration ---
        # 使用一个极小值(epsilon)来处理浮点数精度问题
        while time_remaining_ms > 1e-9:
            current_tempo_event = tempo_list[current_tempo_idx]
            current_bpm = current_tempo_event.tempo

            # 计算在当前BPM下，每个tick持续多少毫秒
            # 1分钟 = 60,000毫秒
            # 每分钟节拍数 = bpm
            # 每拍tick数 = TICK_PER_BEAT
            # ms_per_tick = (毫秒/分钟) / (节拍/分钟) / (tick/节拍) = (60000 / bpm) / TICK_PER_BEAT
            ms_per_tick = (60 * 1000.0 / current_bpm) / 500

            # 确定当前BPM区间的结束点
            # 如果是最后一个tempo，则它会一直持续下去
            end_of_segment_tick = float('inf')
            if current_tempo_idx + 1 < len(tempo_list):
                end_of_segment_tick = tempo_list[current_tempo_idx + 1].time

            # 计算从当前位置到本BPM区间结束，有多少tick
            ticks_in_segment = end_of_segment_tick - current_tick
            # 这些tick总共持续多少毫秒
            ms_in_segment = ticks_in_segment * ms_per_tick

            # --- 步骤4: 判断与更新 ---
            if time_remaining_ms <= ms_in_segment:
                # 如果剩余需要的时间，在本BPM区间内就能满足
                # 计算还需要多少tick来凑够剩余的毫秒数
                ticks_needed = time_remaining_ms / ms_per_tick
                total_ticks_duration += ticks_needed
                # 时间已全部消耗完毕，跳出循环
                time_remaining_ms = 0
            else:
                # 如果本BPM区间的时间不够用
                # 消耗掉整个区间的tick和毫秒数
                total_ticks_duration += ticks_in_segment
                time_remaining_ms -= ms_in_segment
                
                # 更新“指针”，移动到下一个BPM区间的起点
                current_tick = end_of_segment_tick
                current_tempo_idx += 1

        # 返回四舍五入后的总tick数
        return round(total_ticks_duration)

    def ms_to_tick(target_ms, tempo_list):
        # --- 边缘情况处理 ---
        if target_ms <= 0:
            return 0
        if not tempo_list:
            # 如果没有提供tempo信息，则假定为默认的120 BPM
            tempo_list = [TempoChange(120, 0)]

        # --- 步骤1: 初始化累加器 ---
        accumulated_ms = 0.0

        # --- 步骤2: 遍历所有“有终点”的BPM区间 ---
        # 我们遍历到倒数第二个元素，因为每个循环处理的是 tempo[i] 到 tempo[i+1] 的区间
        for i in range(len(tempo_list) - 1):
            current_tempo_event = tempo_list[i]
            next_tempo_event = tempo_list[i+1]

            current_bpm = current_tempo_event.tempo
            
            # 计算当前区间的tick数和对应的毫秒数
            ticks_in_segment = next_tempo_event.time - current_tempo_event.time
            
            # 如果区间长度为0，直接跳过，避免除零错误
            if ticks_in_segment == 0:
                continue
                
            ms_per_tick = (60 * 1000.0 / current_bpm) / 500
            ms_in_segment = ticks_in_segment * ms_per_tick

            # --- 步骤3: 判断目标是否在本区间内 ---
            if target_ms <= accumulated_ms + ms_in_segment:
                # 目标在本区间内！
                ms_into_segment = target_ms - accumulated_ms
                ticks_needed = ms_into_segment / ms_per_tick
                
                # 最终tick = 本区间起始tick + 在本区间内转换出的tick
                final_tick = current_tempo_event.time + ticks_needed
                return round(final_tick)
            
            # 如果目标不在本区间，则累加本区间的总毫秒数，继续下一个循环
            accumulated_ms += ms_in_segment

        # --- 步骤4: 如果循环结束仍未返回，说明目标在最后一个BPM区间内 ---
        last_tempo_event = tempo_list[-1]
        last_bpm = last_tempo_event.tempo
        
        ms_per_tick = (60 * 1000.0 / last_bpm) / 500

        # 计算进入最后一个区间后，还需要多少毫秒
        ms_into_segment = target_ms - accumulated_ms
        ticks_needed = ms_into_segment / ms_per_tick

        # 最终tick = 最后一个区间的起始tick + 剩余毫秒转换的tick
        final_tick = last_tempo_event.time + ticks_needed
        return round(final_tick)

    norm_score = merge_and_sort(score_midi_obj) #normalize_midi(score_midi_obj)
    norm_performance = normalize_midi(performance_midi_obj)
    
    score_notes = norm_score.instruments[0].notes
    performance_notes = norm_performance.instruments[0].notes
    performance_ccs = norm_performance.instruments[0].control_changes

    # print(len(score_notes))
    # print(len(performance_notes))

    start_list = []
    last = -1
    score_start = score_notes[0].start
    performance_start = performance_notes[0].start
    for i in range(len(score_notes)):
        performance_notes[i].end -= performance_start
        performance_notes[i].start -= performance_start
        score_notes[i].end -= score_start
        score_notes[i].start -= score_start
        if score_notes[i].start != last:
            start_list.append((score_notes[i].start, performance_notes[i].start, i))
            last = score_notes[i].start
    
    for i in range(len(performance_ccs)):
        performance_ccs[i].time -= performance_start

    score_interval_list = []
    performance_interval_list = []

    for i in range(len(start_list)-1):
        score_interval_list.append(start_list[i+1][0] - start_list[i][0])
        performance_interval_list.append(start_list[i+1][1] - start_list[i][1])
    #print(start_list)
    #print(score_interval_list)
    #print(performance_interval_list)

    tempo_list = []
    start_note_offset = []
    for i in range(len(score_interval_list)):
        if performance_interval_list[i] != 0:
            bpm = 120.0 / performance_interval_list[i] * score_interval_list[i]
        else:
            bpm = 500

        #if bpm > 300:
        #    start_note_offset.append(300 / 120.0 * performance_interval_list[i] - score_interval_list[i])
        #elif bpm < 10:
        #    start_note_offset.append(10 / 120.0 * performance_interval_list[i] - score_interval_list[i])
        #else:
        start_note_offset.append(0)
        tempo_list.append(max(min(bpm, 500), 10))
        #tempo_list.append(120.0 / performance_interval_list[i] * score_interval_list[i])
    #print(tempo_list)

    for i in range(1, len(start_note_offset)):
        start_note_offset[i] += start_note_offset[i-1]
    #print(start_note_offset)

    #print(len(tempo_list))
    #print(len(start_list))
    note_tempo_list = []
    note_performance_align = []
    note_start_offset = [0]
    cnt = 0
    for i in range(len(score_notes)):
        if cnt < len(start_list) - 2 and i >= start_list[cnt + 1][2]:
            cnt += 1
        note_tempo_list.append(tempo_list[cnt])
        note_performance_align.append(start_list[cnt][1])
        note_start_offset.append(start_note_offset[cnt])
    #print(note_start_offset)

    for i in range(len(score_notes)):
        score_notes[i].start += note_start_offset[i]
    note_interval_list = [0]
    for i in range(len(score_notes)-1):
        note_interval_list.append(score_notes[i+1].start - score_notes[i].start)

    #print(note_tempo_list)
    #print(note_performance_align)

    #for i in range(len(performance_notes)):
        #print(performance_notes[i].start)

    micro_shift_list = [0]
    cnt = 1
    last_time = 0
    for i in range(1, len(score_notes)):
        last_time += note_interval_list[i] / note_tempo_list[i-1] * 120
        micro_shift_list.append((performance_notes[i].start - last_time) / 120 * note_tempo_list[i-1])
        #last_time = note_performance_align[i]
        #print(last_time)

    #print(micro_shift_list)
    #plt.plot(tempo_list)

    res = MidiFile(ticks_per_beat=500)
    res_notes = []
    start_time_list = []
    tempo_list_filter = []
    cc_list = []
    last = -1
    for i in range(len(score_notes)):
        start_time_list.append(round(score_notes[i].start + micro_shift_list[i]))
        #res_notes.append(Note(performance_notes[i].velocity, score_notes[i].pitch, round(score_notes[i].start + micro_shift_list[i]), round(score_notes[i].start + micro_shift_list[i]) + 100))
        #res.tempo_changes.append(TempoChange(round(note_tempo_list[i]), round(score_notes[i].start + micro_shift_list[i])))
        #print(last , round(note_tempo_list[i]))
        if last != round(note_tempo_list[i]):
            tempo_list_filter.append(TempoChange(round(note_tempo_list[i]), round(score_notes[i].start + micro_shift_list[i])))
            last = round(note_tempo_list[i])
    for i in range(len(score_notes)):
        res_notes.append(
            Note(
                performance_notes[i].velocity, 
                score_notes[i].pitch, 
                start_time_list[i], 
                start_time_list[i]+compute_duration(start_time_list[i], performance_notes[i].duration, tempo_list_filter)
            )
        )

    for cc in performance_ccs:
        cc_list.append(ControlChange(64, cc.value, ms_to_tick(cc.time, tempo_list_filter)))
        
    #print(tempo_list_filter)
    res.tempo_changes = tempo_list_filter
    res.instruments.append(Instrument(program=0, is_drum=False, name="Piano", notes=res_notes, control_changes=cc_list))
    res.time_signature_changes = norm_score.time_signature_changes
    res.key_signature_changes = norm_score.key_signature_changes
    return res


if __name__ == "__main__":
    pass