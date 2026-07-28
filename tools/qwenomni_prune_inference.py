import torch
import torch.nn.functional as F
from qwen_omni_utils import process_mm_info
import sys
import numpy as np
import os
from PIL import Image
from typing import List

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from tools.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from tools.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration

DEFAULT_MAX_PIXELS = 128 * 28 * 28


class ClipScorer:

    def __init__(self, model_name: str = None, device: str = "cuda", cache_image_features: bool = True):
        self.method = "clip"
        self.device = device
        self.cache_image_features = cache_image_features
        self._cached_image_features = None
        self._cached_video_id = None
        self._init_clip(model_name or "openai/clip-vit-large-patch14")

    def _init_clip(self, model_name: str):
        from transformers import CLIPModel, CLIPTokenizerFast, CLIPImageProcessor

        self.model = CLIPModel.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        self.model.eval()
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        try:
            self.tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
        except (ValueError, Exception):
            try:
                self.tokenizer = CLIPTokenizerFast.from_pretrained(model_name, from_slow=True)
            except Exception:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _encode_images(self, frames: List[Image.Image]) -> torch.Tensor:
        batch_size = 32
        all_features = []
        with torch.no_grad():
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i + batch_size]
                inputs = self.image_processor(images=batch_frames, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                features = self.model.get_image_features(**inputs)
                features = F.normalize(features.float(), dim=-1)
                all_features.append(features)
        return torch.cat(all_features, dim=0)

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            features = self.model.get_text_features(**inputs)
            features = F.normalize(features.float(), dim=-1)
        return features

    def _extract_frames_from_video_tensor(self, cached_videos: list) -> List[Image.Image]:
        if cached_videos is None or len(cached_videos) == 0:
            return []
        video_tensor = cached_videos[0]
        if isinstance(video_tensor, torch.Tensor):
            video_np = video_tensor.cpu().numpy()
        else:
            video_np = np.array(video_tensor)
        if video_np.ndim == 4:
            if video_np.shape[1] == 3:
                video_np = video_np.transpose(0, 2, 3, 1)
        if video_np.max() <= 1.0:
            video_np = (video_np * 255).astype(np.uint8)
        else:
            video_np = video_np.astype(np.uint8)
        frames = [Image.fromarray(video_np[i]) for i in range(video_np.shape[0])]
        return frames

    def _extract_frames_from_video_path(self, video_path: str, frames_num: int) -> List[Image.Image]:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(), num_threads=1)
        total_frames = len(vr)
        indices = np.linspace(0, total_frames - 1, frames_num, dtype=int)
        frames_np = vr.get_batch(indices).asnumpy()
        frames = [Image.fromarray(frames_np[i]) for i in range(frames_np.shape[0])]
        return frames

    def compute_frame_scores_batch(
        self,
        questions: List[str],
        video_path: str = None,
        cached_videos: list = None,
        frames_num: int = 128,
        video_id: str = None,
        top_k: int = 1,
    ) -> List[List[float]]:
        if self.cache_image_features and video_id and video_id == self._cached_video_id:
            image_features = self._cached_image_features
        else:
            if cached_videos is not None:
                frames = self._extract_frames_from_video_tensor(cached_videos)
            elif video_path is not None:
                frames = self._extract_frames_from_video_path(video_path, frames_num)
            else:
                raise ValueError("Must provide either video_path or cached_videos")
            if len(frames) == 0:
                return [[1.0] for _ in questions]
            image_features = self._encode_images(frames)
            if self.cache_image_features and video_id:
                self._cached_image_features = image_features
                self._cached_video_id = video_id

        num_frames = image_features.shape[0]

        valid_questions = [q for q in questions if q and q.strip()]
        if not valid_questions:
            return [[1.0] * num_frames for _ in questions]

        text_features = self._encode_texts(valid_questions)

        all_frame_scores = []
        valid_idx = 0
        for question in questions:
            if not question or not question.strip():
                all_frame_scores.append([1.0] * num_frames)
                continue
            with torch.no_grad():
                sims = (image_features @ text_features[valid_idx].unsqueeze(-1)).squeeze(-1)
                frame_scores = sims.cpu().tolist()
            valid_idx += 1
            all_frame_scores.append(frame_scores)

        return all_frame_scores


model = None
processor = None


def init_model(model_path: str, device_map="auto", torch_dtype=torch.bfloat16):
    global model, processor
    if model is not None:
        return model, processor
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    return model, processor

def get_cached_video_embeds(video_path, frames_num, max_pixels=None):
    if max_pixels is None:
        max_pixels = DEFAULT_MAX_PIXELS
    USE_AUDIO_IN_VIDEO = True

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "max_frames": frames_num, "fps": 2, "max_pixels": max_pixels},
            {"type": "text", "text": ""}
        ]},
    ]

    audios, images, videos, video_kwargs = process_mm_info(
        conversation,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
        return_video_kwargs=True
    )

    if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
        video_kwargs["fps"] = video_kwargs["fps"][0]
    video_kwargs["use_audio_in_video"] = USE_AUDIO_IN_VIDEO

    re_video_kwargs = video_kwargs.copy()

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, videos_kwargs=video_kwargs,
    )

    pixel_values_videos = inputs["pixel_values_videos"].to(model.device)
    video_grid_thw = inputs["video_grid_thw"].to(model.device)

    with torch.no_grad():
        video_embeds, _ = model.thinker.visual(
            pixel_values_videos.type(model.thinker.visual.dtype),
            grid_thw=video_grid_thw
        )

    return video_embeds, video_grid_thw, audios, videos, re_video_kwargs


def get_cached_audio_embeds(audios):
    if audios is None or len(audios) == 0:
        return None, None

    audio_inputs = processor.feature_extractor(
        audios, sampling_rate=16000, return_tensors="pt",
        return_attention_mask=True, padding="max_length",
    )

    input_features = audio_inputs["input_features"].to(model.device).to(model.dtype)
    feature_attention_mask = audio_inputs["attention_mask"].to(model.device)

    audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
    input_features_processed = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)

    audio_feat_lengths, audio_output_lengths = model.thinker.audio_tower._get_feat_extract_output_lengths(
        audio_feature_lengths
    )

    with torch.no_grad():
        audio_outputs = model.thinker.audio_tower(
            input_features_processed,
            feature_lens=audio_feature_lengths,
            aftercnn_lens=audio_feat_lengths,
        )
        audio_embeds = audio_outputs.last_hidden_state

    return audio_embeds, audio_output_lengths


def compute_audio_semantic_scores_batch(audio_embeds, questions, tokens_per_second=25, top_k=5, tau_agg=0.05):
    if audio_embeds is None:
        return [[1.0] for _ in questions]

    num_tokens = audio_embeds.shape[0]
    num_seconds = num_tokens // tokens_per_second

    if num_seconds == 0:
        return [[1.0] for _ in questions]

    audio_norm = F.normalize(audio_embeds.float(), dim=-1)
    embed_tokens = model.thinker.get_input_embeddings()
    eps = 1e-6

    scores_list = []

    for question in questions:
        if not question or not question.strip():
            scores_list.append([1.0] * num_seconds)
            continue

        with torch.no_grad():
            tokens = processor.tokenizer(question, return_tensors="pt", add_special_tokens=False)
            token_ids = tokens.input_ids.to(model.device)
            text_embeds = embed_tokens(token_ids).squeeze(0)

            token_norms = text_embeds.norm(dim=-1)
            max_norm = token_norms.max() + eps
            gates = token_norms / max_norm

            text_norm = F.normalize(text_embeds.float(), dim=-1)

            S_cross = audio_norm @ text_norm.T
            gated_S_cross = S_cross * gates.unsqueeze(0)
            weights = torch.softmax(gated_S_cross / tau_agg, dim=-1)
            s_text = (weights * S_cross).sum(dim=-1)

        scores = []
        for sec_idx in range(num_seconds):
            start_idx = sec_idx * tokens_per_second
            end_idx = start_idx + tokens_per_second
            sec_token_scores = s_text[start_idx:end_idx]
            k = min(top_k, len(sec_token_scores))
            top_values = sec_token_scores.topk(k).values
            scores.append(top_values.mean().item())

        scores_list.append(scores)

    return scores_list


def compute_audio_tokens_allocation(audio_semantic_scores, total_budget, min_tokens=3, max_tokens=25):
    if audio_semantic_scores is None or len(audio_semantic_scores) == 0:
        return np.array([])

    scores = np.array(audio_semantic_scores, dtype=np.float64)
    num_seconds = len(scores)

    scores = np.maximum(scores, 0.0)
    if scores.sum() < 1e-12:
        scores = np.ones(num_seconds, dtype=np.float64)

    total_budget = float(np.clip(total_budget, num_seconds * min_tokens, num_seconds * max_tokens))
    allocation = scores / scores.sum() * total_budget
    locked = np.zeros(num_seconds, dtype=bool)

    for _ in range(10):
        allocation = np.clip(allocation, min_tokens, max_tokens)
        clipped_total = allocation.sum()
        diff = clipped_total - total_budget

        if abs(diff) < 1e-6:
            break

        if diff > 0:
            newly_locked = (allocation <= min_tokens) & (~locked)
        else:
            newly_locked = (allocation >= max_tokens) & (~locked)

        if not newly_locked.any():
            break

        locked[newly_locked] = True
        if locked.all():
            break

        locked_total = allocation[locked].sum()
        remaining_budget = total_budget - locked_total

        unlocked_scores = scores[~locked]
        if unlocked_scores.sum() < 1e-12:
            allocation[~locked] = remaining_budget / (~locked).sum()
        else:
            allocation[~locked] = unlocked_scores / unlocked_scores.sum() * remaining_budget

    int_budget = int(np.round(total_budget))
    floors = np.floor(allocation).astype(np.int32)
    remainders = allocation - floors

    deficit = int_budget - floors.sum()
    if deficit > 0:
        indices = np.argsort(-remainders)
        for i in range(min(deficit, num_seconds)):
            floors[indices[i]] += 1
    elif deficit < 0:
        indices = np.argsort(remainders)
        for i in range(min(-deficit, num_seconds)):
            floors[indices[i]] -= 1

    tokens_per_second = np.clip(floors, min_tokens, max_tokens)
    return tokens_per_second


def compute_audio_compress_softmax(audio_semantic_scores, total_budget, min_tokens=3, max_tokens=25,
                                   temperature=None, target_avg_tokens=None):
    return compute_audio_tokens_allocation(
        audio_semantic_scores, total_budget=total_budget,
        min_tokens=min_tokens, max_tokens=max_tokens,
    )


def qwen_prune_inference_with_cache(video_path, query, score_list, frames_num,
                                     cached_video_embeds, cached_video_grid_thw,
                                     cached_audios, cached_videos, cached_video_kwargs,
                                     video_duration=None,
                                     audio_semantic_scores=None,
                                     visual_prune_ratio=0.8,
                                     audio_prune_ratio=0.35,
                                     max_pixels=None,
                                     audio_coverage_ratio=1.0,
                                     cached_audio_embeds=None,
                                     cached_audio_output_lengths=None,
                                     ):
    if max_pixels is None:
        max_pixels = DEFAULT_MAX_PIXELS
    USE_AUDIO_IN_VIDEO = True

    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "max_frames": frames_num, "fps": 2, "max_pixels": max_pixels},
                {"type": "text", "text": query},
            ],
        },
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    input_video_kwargs = cached_video_kwargs.copy()

    if score_list is not None:
        score_list = torch.tensor(score_list, device=model.device) if not isinstance(score_list, torch.Tensor) else score_list

    use_audio_cache = (cached_audio_embeds is not None and cached_audio_output_lengths is not None)

    if use_audio_cache:
        _tmp_original_audio_length = cached_audio_output_lengths[0].item() if isinstance(cached_audio_output_lengths, torch.Tensor) else cached_audio_output_lengths
    elif cached_audios is not None and len(cached_audios) > 0:
        _tmp_audio_inputs = processor.feature_extractor(
            cached_audios, sampling_rate=16000, return_tensors="pt",
            return_attention_mask=True, padding="max_length",
        )
        _tmp_feat_len = _tmp_audio_inputs["attention_mask"].sum(-1)
        _tmp_input_lengths = (_tmp_feat_len - 1) // 2 + 1
        _tmp_original_audio_length = ((_tmp_input_lengths - 2) // 2 + 1)[0].item()
    else:
        _tmp_original_audio_length = 0

    audio_compress_mask = None

    if audio_prune_ratio > 0 and _tmp_original_audio_length > 0 and audio_semantic_scores is not None:
        tokens_per_second = 25
        num_audio_seconds_precise = _tmp_original_audio_length // tokens_per_second
        tail_tokens = _tmp_original_audio_length - num_audio_seconds_precise * tokens_per_second

        num_audio_seconds_scores = len(audio_semantic_scores)
        num_audio_seconds = min(num_audio_seconds_precise, num_audio_seconds_scores)

        if num_audio_seconds_scores > num_audio_seconds:
            effective_scores_aligned = audio_semantic_scores[:num_audio_seconds]
        elif num_audio_seconds_scores < num_audio_seconds:
            effective_scores_aligned = list(audio_semantic_scores) + [1.0] * (num_audio_seconds - num_audio_seconds_scores)
        else:
            effective_scores_aligned = audio_semantic_scores

        total_original_full_secs = num_audio_seconds * tokens_per_second
        total_budget_full_secs = total_original_full_secs * (1.0 - audio_prune_ratio)

        audio_compress_mask = compute_audio_compress_softmax(
            audio_semantic_scores=effective_scores_aligned,
            total_budget=total_budget_full_secs,
            min_tokens=10,
            max_tokens=tokens_per_second,
        )

    if use_audio_cache:
        inputs = processor(
            text=text, audio=None, images=None, videos=cached_videos,
            return_tensors="pt", padding=True,
            importance_scores=score_list,
            target_prune_ratio=1.0 - visual_prune_ratio,
            audio_compress_mask=audio_compress_mask,
            cached_audio_length=_tmp_original_audio_length,
            videos_kwargs=input_video_kwargs,
        )
    else:
        inputs = processor(
            text=text, audio=cached_audios, images=None, videos=cached_videos,
            return_tensors="pt", padding=True,
            importance_scores=score_list,
            target_prune_ratio=1.0 - visual_prune_ratio,
            audio_compress_mask=audio_compress_mask,
            videos_kwargs=input_video_kwargs,
        )

    _audio_compress_mask = inputs.get("audio_compress_mask", None)
    _audio_token_indices = inputs.get("audio_token_indices", None)

    if "pixel_values_videos" in inputs:
        del inputs["pixel_values_videos"]
    if "audio_compress_mask" in inputs:
        del inputs["audio_compress_mask"]
    if "audio_token_indices" in inputs:
        del inputs["audio_token_indices"]

    if use_audio_cache:
        if "input_features" in inputs:
            del inputs["input_features"]
        if "feature_attention_mask" in inputs:
            del inputs["feature_attention_mask"]

    inputs = inputs.to(model.device).to(model.dtype)

    generate_kwargs = dict(
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
        do_sample=False, return_audio=False,
        score_list=score_list,
        target_prune_ratio=1.0 - visual_prune_ratio,
        cached_video_embeds=cached_video_embeds,
        cached_video_grid_thw=cached_video_grid_thw,
        audio_compress_mask=_audio_compress_mask,
        audio_token_indices=_audio_token_indices,
    )

    if use_audio_cache:
        generate_kwargs["cached_audio_embeds"] = cached_audio_embeds
        generate_kwargs["audio_feature_lengths"] = torch.tensor(
            [_tmp_original_audio_length], device=model.device, dtype=torch.long
        )

    text_ids = model.generate(
        **inputs,
        **generate_kwargs,
    )

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, text_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    result = output_text[0][0] if isinstance(output_text[0], list) else output_text[0]

    del inputs, text_ids, generated_ids_trimmed, output_text
    if _audio_compress_mask is not None:
        del _audio_compress_mask
    if _audio_token_indices is not None:
        del _audio_token_indices
    torch.cuda.empty_cache()

    return result