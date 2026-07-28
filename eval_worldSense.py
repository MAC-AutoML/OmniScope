import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import gc
import torch
from decord import VideoReader, cpu
import numpy as np
import json
from tqdm import tqdm

from tools.qwenomni_prune_inference import (
    init_model,
    get_cached_video_embeds,
    qwen_prune_inference_with_cache,
    get_cached_audio_embeds,
    compute_audio_semantic_scores_batch,
    ClipScorer,
)

QWEN_MODEL_PATH = "/path/to/Qwen2.5-Omni-7B"
CLIP_MODEL_NAME = "/path/to/clip-vit-large-patch14-336"
data_path = "/path/to/worldsense_videos"

frames_num = 128
topk = 10

VISUAL_PRUNE_RATIO = 0.6
AUDIO_PRUNE_RATIO = 0.25

AUDIO_BOOST_SELF = 1.2
AUDIO_BOOST_NEIGHBOR = 1.5
AUDIO_BOOST_RADIUS = 3
AUDIO_BOOST_DECAY = 0.5
BOOST_PERCENTILE = 80

MAX_PIXELS = 768 * 28 * 28
SAVE_SCORE_CACHE = False
SAME_WEIGHT = False

file_name = f"worldsense_{frames_num}_{VISUAL_PRUNE_RATIO}_{AUDIO_PRUNE_RATIO}_px{MAX_PIXELS // (28*28)}_topk_{topk}_sameW_{SAME_WEIGHT}"
json_file = f"results/worldSense/{file_name}.json"
score_cache_file = f"results/worldSense/clip_qwen_scores_cache.json"
rep_list = []
score_data = {}

init_model(QWEN_MODEL_PATH)
_clip_scorer = None

os.makedirs("results/worldSense", exist_ok=True)

with open("data/worldsense_format.json", 'r', encoding='utf-8') as file:
    mme_data = json.load(file)

if os.path.exists(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        rep_list = json.load(file)

if os.path.exists(score_cache_file):
    with open(score_cache_file, 'r', encoding='utf-8') as file:
        score_data = json.load(file)

def get_clip_scorer():
    global _clip_scorer
    if _clip_scorer is None:
        _clip_scorer = ClipScorer(
            model_name=CLIP_MODEL_NAME,
            device="cuda",
            cache_image_features=True,
        )
    return _clip_scorer


def temporal_merge_scores(scores_per_question, merge_size=2):
    merged_all = []
    for scores in scores_per_question:
        merged = []
        for i in range(0, len(scores) - merge_size + 1, merge_size):
            chunk = scores[i:i + merge_size]
            merged.append(sum(chunk) / len(chunk))
        remainder = len(scores) % merge_size
        if remainder > 0:
            merged.append(sum(scores[-remainder:]) / remainder)
        merged_all.append(merged)
    return merged_all


def apply_neighbor_boost(scores, percentile=80, self_boost=1.2, neighbor_boost=1.5, radius=3, decay=0.5):
    if len(scores) <= 1:
        return scores

    scores_arr = np.array(scores, dtype=float)
    threshold = np.percentile(scores_arr, percentile)
    high_indices = np.where(scores_arr >= threshold)[0]

    multipliers = np.ones_like(scores_arr)

    for idx in high_indices:
        multipliers[idx] = max(multipliers[idx], self_boost)

    for idx in high_indices:
        for d in range(1, radius + 1):
            factor = 1.0 + (neighbor_boost - 1.0) * (decay ** d)
            for neighbor in [idx - d, idx + d]:
                if 0 <= neighbor < len(scores_arr):
                    multipliers[neighbor] = max(multipliers[neighbor], factor)

    return (scores_arr * multipliers).tolist()



index = len(rep_list)
for item in tqdm(mme_data[index:], desc="Processing items"):
    video_path = os.path.join(data_path, item['url'] + ".mp4")
    content = item.copy()

    vr = VideoReader(video_path, ctx=cpu(), num_threads=1)
    video_time = len(vr) / vr.get_avg_fps()
    frames_num_input = frames_num + (frames_num % 2)
    frames_num_input = min(frames_num_input, len(vr) - (len(vr) % 2))

    cached_video_embeds, cached_video_grid_thw, cached_audios, cached_videos, cached_video_kwargs = \
        get_cached_video_embeds(video_path, frames_num_input, max_pixels=MAX_PIXELS)

    audio_embeds, audio_output_lengths = get_cached_audio_embeds(cached_audios)

    query_list = []
    for question in content['questions']:
        qs = question['question'] + "\n" + " ".join(question['options'])
        query_list.append(qs)

    video_id = item['url']
    if video_id in score_data and 'visual' in score_data[video_id]:
        visual_score_list = score_data[video_id]['visual']
    else:
        scorer = get_clip_scorer()

        raw_visual_scores = scorer.compute_frame_scores_batch(
            query_list,
            video_path=video_path,
            cached_videos=cached_videos,
            frames_num=frames_num_input,
            video_id=item['url'],
        )

        visual_score_list = temporal_merge_scores(raw_visual_scores, merge_size=2)

        T_qwen = cached_video_grid_thw[0][0].item()

        for q_idx in range(len(visual_score_list)):
            if len(visual_score_list[q_idx]) != T_qwen:
                if len(visual_score_list[q_idx]) > T_qwen:
                    visual_score_list[q_idx] = visual_score_list[q_idx][:T_qwen]
                else:
                    visual_score_list[q_idx] += [1.0] * (T_qwen - len(visual_score_list[q_idx]))

    audio_score_list = compute_audio_semantic_scores_batch(audio_embeds, query_list)

    score_data[item['url']] = {
        'visual': visual_score_list,
        'audio': audio_score_list,
    }

    for q_num, question in enumerate(content['questions']):
        visual_score = visual_score_list[q_num]
        visual_score = [max(0, w * 100) ** 3 for w in visual_score]

        audio_semantic_scores = [max(0, w * 10) ** 2 for w in audio_score_list[q_num]]

        audio_semantic_scores = apply_neighbor_boost(
            audio_semantic_scores,
            percentile=BOOST_PERCENTILE,
            self_boost=AUDIO_BOOST_SELF,
            neighbor_boost=AUDIO_BOOST_NEIGHBOR,
            radius=AUDIO_BOOST_RADIUS,
            decay=AUDIO_BOOST_DECAY,
        )

        if SAME_WEIGHT:
            visual_score = [1.0] * len(visual_score)
            audio_semantic_scores = [1.0] * len(audio_semantic_scores)

        qs = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option. Question: " + question['question'] + '\n' + " ".join(question['options']) + '\nThe best answer is:'

        res = qwen_prune_inference_with_cache(
            video_path, qs, visual_score, frames_num_input,
            cached_video_embeds, cached_video_grid_thw,
            cached_audios, cached_videos, cached_video_kwargs,
            video_duration=video_time,
            audio_semantic_scores=audio_semantic_scores,
            visual_prune_ratio=VISUAL_PRUNE_RATIO,
            audio_prune_ratio=AUDIO_PRUNE_RATIO,
            max_pixels=MAX_PIXELS,
            cached_audio_embeds=audio_embeds,
            cached_audio_output_lengths=audio_output_lengths,
        )

        question['response'] = res

    del cached_video_embeds, cached_video_grid_thw
    del audio_embeds, audio_output_lengths
    if 'cached_audios' in dir():
        del cached_audios
    if 'cached_videos' in dir():
        del cached_videos
    if 'cached_video_kwargs' in dir():
        del cached_video_kwargs
    gc.collect()
    torch.cuda.empty_cache()

    rep_list.append(content)
    index += 1

    with open(json_file, "w", encoding='utf-8') as file:
        json.dump(rep_list, file, ensure_ascii=False, indent=4)

    if SAVE_SCORE_CACHE:
        with open(score_cache_file, "w", encoding='utf-8') as file:
            json.dump(score_data, file, ensure_ascii=False, indent=4)