# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for Qwen2.5Omni.
"""
import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="librosa"
)


import logging

import torch
import re
from typing import List, Optional, Union

import numpy as np

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs
from transformers.tokenization_utils_base import AudioInput, PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput, make_batched_videos
import os
import sys




AUDIO_TOKENS_PER_SECOND = 25
AUDIO_COMPRESSED_TOKENS_PER_SECOND = 15

class Qwen2_5_OmniVideosKwargs(VideosKwargs):
    fps: Optional[List[int]] = None
    use_audio_in_video: Optional[bool] = None
    seconds_per_chunk: Optional[float] = None
    position_id_per_seconds: Optional[int] = None
    min_pixels: Optional[int]
    max_pixels: Optional[int]
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]


class Qwen2_5_OmniImagesKwargs(ImagesKwargs):
    min_pixels: Optional[int]
    max_pixels: Optional[int]
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]


class Qwen2_5OmniProcessorKwargs(ProcessingKwargs, total=False):
    videos_kwargs: Qwen2_5_OmniVideosKwargs
    images_kwargs: Qwen2_5_OmniImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "padding_side": "left",
        },
        "videos_kwargs": {
            "seconds_per_chunk": 2.0,
            "position_id_per_seconds": 25,
            "use_audio_in_video": False,
            "min_pixels": 128 * 28 * 28,
            "max_pixels": 768 * 28 * 28,
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": "max_length",
            "return_attention_mask": True,
        },
    }


class Qwen2_5OmniProcessor(ProcessorMixin):
    r"""
    Constructs a Qwen2.5Omni processor.
    [`Qwen2_5OmniProcessor`] offers all the functionalities of [`Qwen2VLImageProcessor`], [`WhisperFeatureExtractor`], and [`Qwen2TokenizerFast`]. See the
    [`~Qwen2_5OmniProcessor.__call__`] and [`~Qwen2_5OmniProcessor.decode`] for more information.

    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor.
        video_processor ([`Qwen2VLVideoProcessor`], *optional*):
            The video processor.
        feature_extractor ([`WhisperFeatureExtractor`], *optional*):
            The audio feature extractor.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The text tokenizer.
        chat_template (`Optional[str]`, *optional*):
            The Jinja template to use for formatting the conversation. If not provided, the default chat template is used.
    """

    attributes = ["image_processor", "video_processor", "feature_extractor", "tokenizer"]
    image_processor_class = "Qwen2VLImageProcessor"
    video_processor_class = "Qwen2VLVideoProcessor"
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
    valid_kwargs = ["chat_template"]

    def __init__(
        self, image_processor=None, video_processor=None, feature_extractor=None, tokenizer=None, chat_template=None
    ):
        super().__init__(image_processor, video_processor, feature_extractor, tokenizer, chat_template=chat_template)
        self.image_token = self.tokenizer.image_token
        self.audio_token = self.tokenizer.audio_token
        self.video_token = self.tokenizer.video_token
        self.vision_bos_token = self.tokenizer.vision_bos_token
        self.vision_eos_token = self.tokenizer.vision_eos_token
        self.audio_bos_token = self.tokenizer.audio_bos_token
        self.audio_eos_token = self.tokenizer.audio_eos_token

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        videos: VideoInput = None,
        audio: AudioInput = None,
        importance_scores=None, 
        target_prune_ratio=None,
        audio_compress_mask=None,
        cached_audio_length=None,
        **kwargs: Unpack[Qwen2_5OmniProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the audio(s), this method forwards the `audio` and `kwargs` arguments to
        WhisperFeatureExtractor's [`~WhisperFeatureExtractor.__call__`] if `audio` is not `None`. To prepare the vision inputs,
        this method forwards the `vision_infos` and `kwargs` arguments to Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`]
        if `vision_infos` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
            audio (`np.ndarray`, `List[np.ndarray]`):
                The audio or batch of audio to be prepared. Each audio can be a NumPy array.
            cached_audio_length (`int`, *optional*):
                Pre-computed original audio token length (from audio_tower output).
                When provided with audio=None, skips feature extraction but still generates
                correct audio token placeholders for the input sequence.
        """

        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")

        output_kwargs = self._merge_kwargs(
            Qwen2_5OmniProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        seconds_per_chunk = output_kwargs["videos_kwargs"].pop("seconds_per_chunk")
        position_id_per_seconds = output_kwargs["videos_kwargs"].pop("position_id_per_seconds")
        use_audio_in_video = output_kwargs["videos_kwargs"].pop("use_audio_in_video")
        fps = output_kwargs["videos_kwargs"].pop("fps", 2.0)


        _audio_compress_mask = None
        _audio_token_indices = None
        
        if audio is not None:
            output_kwargs["audio_kwargs"]["padding"] = "max_length"  # Support "max_length" padding only here
            audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
            audio_inputs["feature_attention_mask"] = audio_inputs.pop(
                "attention_mask"
            )  # rename feature_attention_mask to prevent conflicts later on
            audio_inputs["input_features"] = audio_inputs.pop(
                "input_features"
            )  # rename input_features to prevent conflicts later on
            input_lengths = (audio_inputs["feature_attention_mask"].sum(-1) - 1) // 2 + 1
            original_audio_lengths = (input_lengths - 2) // 2 + 1
            
 
            if audio_compress_mask is not None and len(audio_compress_mask) > 0:
                original_len = original_audio_lengths[0].item()  
                _audio_token_indices, new_audio_length = self.compute_audio_token_indices(
                    original_len, audio_compress_mask
                )
                
                
                _audio_compress_mask = audio_compress_mask
                audio_lengths = iter([new_audio_length])
            else:
                audio_lengths = iter(original_audio_lengths.tolist())
        elif cached_audio_length is not None:
            audio_inputs = {}
            original_len = cached_audio_length

            if audio_compress_mask is not None and len(audio_compress_mask) > 0:
                _audio_token_indices, new_audio_length = self.compute_audio_token_indices(
                    original_len, audio_compress_mask
                )
                _audio_compress_mask = audio_compress_mask
                audio_lengths = iter([new_audio_length])
            else:
                audio_lengths = iter([original_len])
        else:
            audio_inputs = {}
            audio_lengths = iter([])

        if images is not None:
            images_inputs = self.image_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
            image_grid_thw = iter(images_inputs["image_grid_thw"])
        else:
            images_inputs = {}
            image_grid_thw = iter([])
        if videos is not None:
            videos = make_batched_videos(videos)
            videos_inputs = self.video_processor(images=None, videos=videos, **output_kwargs["videos_kwargs"])
            fps = [fps] * len(videos)

            videos_inputs["video_second_per_grid"] = [
                self.video_processor.temporal_patch_size / fps[i] for i in range(len(fps))
            ]
            video_grid_thw = iter(videos_inputs["video_grid_thw"])

            video_second_per_grid = iter(videos_inputs["video_second_per_grid"])
        else:
            videos_inputs = {}
            video_grid_thw = iter([])
            video_second_per_grid = iter([])

        if not isinstance(text, list):
            text = [text]
            

        if importance_scores is not None :

            text = self.replace_multimodal_special_tokens_pruning(
                text,
                audio_lengths,
                image_grid_thw,
                video_grid_thw,
                video_second_per_grid=video_second_per_grid,
                use_audio_in_video=use_audio_in_video,
                position_id_per_seconds=position_id_per_seconds,
                seconds_per_chunk=seconds_per_chunk,

                importance_scores=importance_scores,  
                target_prune_ratio=target_prune_ratio,
                audio_token_indices=_audio_token_indices,  
            )
        else:
            text = self.replace_multimodal_special_tokens(
                text,
                audio_lengths,
                image_grid_thw,
                video_grid_thw,
                video_second_per_grid=video_second_per_grid,
                use_audio_in_video=use_audio_in_video,
                position_id_per_seconds=position_id_per_seconds,
                seconds_per_chunk=seconds_per_chunk,
            )

        texts_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        batch_feature = BatchFeature(
            data={**texts_inputs, **images_inputs, **videos_inputs, **audio_inputs},
            tensor_type=kwargs.get("return_tensors"),
        )

        batch_feature["audio_compress_mask"] = _audio_compress_mask
        batch_feature["audio_token_indices"] = _audio_token_indices
        return batch_feature
    def replace_multimodal_special_tokens(
        self,
        text,
        audio_lengths,
        image_grid_thw,
        video_grid_thw,
        video_second_per_grid,
        use_audio_in_video,
        position_id_per_seconds,
        seconds_per_chunk,
    ):
        # Extend mm token length
        merge_length_image = self.image_processor.merge_size**2
        merge_length_video = self.video_processor.merge_size**2

        processed_text = []
        for sample in text:
            positions = []
            special_tokens = [re.escape(tok) for tok in [self.audio_token, self.image_token, self.video_token]]
            pattern = "|".join(special_tokens)
            positions = sorted([(match.start(), match.group()) for match in re.finditer(pattern, sample)])
            positions.sort(key=lambda x: x[0])

            for _, special_token in positions:
                if special_token == self.audio_token:
                    sample = sample.replace(self.audio_token, "<|audio_placeholder|>" * next(audio_lengths), 1)
                elif special_token == self.image_token:
                    image_seq_length = next(image_grid_thw).prod() // merge_length_image
                    sample = sample.replace(self.image_token, "<|image_placeholder|>" * image_seq_length, 1)
                elif special_token == self.video_token:
                    if not use_audio_in_video:
                        video_seq_length = next(video_grid_thw).prod() // merge_length_video
                        sample = sample.replace(self.video_token, "<|video_placeholder|>" * video_seq_length, 1)
                    else:
                        audio_token_indices = np.arange(next(audio_lengths))
                        curr_video_grid_thw = next(video_grid_thw)
                        height = curr_video_grid_thw[1] // self.video_processor.merge_size
                        width = curr_video_grid_thw[2] // self.video_processor.merge_size
                        video_token_indices = np.arange(curr_video_grid_thw[0]).reshape(-1, 1, 1)
                        video_token_indices = np.broadcast_to(
                            video_token_indices, (video_token_indices.shape[0], height, width)
                        ).reshape(-1)
                        video_token_indices = (
                            video_token_indices * next(video_second_per_grid) * position_id_per_seconds
                        )

                        tokens_per_chunk = int(position_id_per_seconds * seconds_per_chunk)
                        
                        video_chunk_indexes = self.get_chunked_index(video_token_indices, tokens_per_chunk)
                        audio_chunk_indexes = self.get_chunked_index(audio_token_indices, tokens_per_chunk)
                        

                        placeholder_string = self.vision_bos_token + self.audio_bos_token
                        for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                            if j < len(video_chunk_indexes):
                                video_seq_length = video_chunk_indexes[j][1] - video_chunk_indexes[j][0]
                                placeholder_string += "<|video_placeholder|>" * video_seq_length
                            if j < len(audio_chunk_indexes):
                                audio_seq_length = audio_chunk_indexes[j][1] - audio_chunk_indexes[j][0]
                                placeholder_string += "<|audio_placeholder|>" * audio_seq_length
                        placeholder_string += self.audio_eos_token + self.vision_eos_token
                        sample = sample.replace(
                            self.vision_bos_token + self.video_token + self.vision_eos_token,
                            placeholder_string,
                            1,
                        )
                        

            sample = sample.replace("<|audio_placeholder|>", self.audio_token)
            sample = sample.replace("<|image_placeholder|>", self.image_token)
            sample = sample.replace("<|video_placeholder|>", self.video_token)

            processed_text.append(sample)
        return processed_text


    def replace_multimodal_special_tokens_pruning(
        self,
        text,
        audio_lengths,
        image_grid_thw,
        video_grid_thw,
        video_second_per_grid,
        use_audio_in_video,
        position_id_per_seconds,
        seconds_per_chunk,
        importance_scores,  
        target_prune_ratio,
        audio_token_indices=None,  
    ):

        merge_length_image = self.image_processor.merge_size**2
        merge_length_video = self.video_processor.merge_size**2

        processed_text = []
        for sample in text:
            positions = []
            special_tokens = [re.escape(tok) for tok in [self.audio_token, self.image_token, self.video_token]]
            pattern = "|".join(special_tokens)
            positions = sorted([(match.start(), match.group()) for match in re.finditer(pattern, sample)])
            positions.sort(key=lambda x: x[0])

            for _, special_token in positions:
                if special_token == self.audio_token:
                    sample = sample.replace(self.audio_token, "<|audio_placeholder|>" * next(audio_lengths), 1)
                elif special_token == self.image_token:
                    image_seq_length = next(image_grid_thw).prod() // merge_length_image
                    sample = sample.replace(self.image_token, "<|image_placeholder|>" * image_seq_length, 1)
                elif special_token == self.video_token:
                    if not use_audio_in_video:
                        video_seq_length = next(video_grid_thw).prod() // merge_length_video
                        sample = sample.replace(self.video_token, "<|video_placeholder|>" * video_seq_length, 1)
                    else:
    
                        audio_len = next(audio_lengths)
                        if audio_token_indices is not None:
                            local_audio_token_indices = audio_token_indices
                        else:
                            local_audio_token_indices = np.arange(audio_len)
                        
                        curr_video_grid_thw = next(video_grid_thw)
                        T = curr_video_grid_thw[0]
                        height = curr_video_grid_thw[1] // self.video_processor.merge_size
                        width = curr_video_grid_thw[2] // self.video_processor.merge_size
                    
    
                        source_num_tokens = T * height * width
                        total_target = target_prune_ratio * source_num_tokens
                        importance_scores = torch.round(importance_scores * 1000) / 1000

                        scores_np = importance_scores.cpu().numpy().astype(np.float64)
                        allocation = scores_np / scores_np.sum() * float(total_target)
                        target_tokens_per_group = torch.tensor(np.floor(allocation + 0.5).astype(np.int64), device=importance_scores.device)
                        debug_tokens_per_frame = []
                        aspect_ratio = float(width) / float(height)
                        video_token_indices = []
                        total_token = 0
                        for i in range(T):
                            cur_token = target_tokens_per_group[i].item()
                            n_tokens_per_group_orig = int(height) * int(width)
                            cur_token = min(cur_token, n_tokens_per_group_orig)
                            target_height = round((cur_token / aspect_ratio) ** 0.5)
                            target_width = round(target_height * aspect_ratio)
                            target_height = max(2, min(target_height, int(height)))
                            target_width = max(2, min(target_width, int(width)))
                            n_tokens = target_height * target_width
                            debug_tokens_per_frame.append(n_tokens)
                            video_token_indices.extend([i] * n_tokens)
                            total_token += n_tokens
                        
                        video_token_indices = np.array(video_token_indices)               
                        video_token_indices = (
                            video_token_indices * next(video_second_per_grid) * position_id_per_seconds
                        )

                        tokens_per_chunk = int(position_id_per_seconds * seconds_per_chunk)
                        video_chunk_indexes = self.get_chunked_index(video_token_indices, tokens_per_chunk)
                        audio_chunk_indexes = self.get_chunked_index(local_audio_token_indices, tokens_per_chunk)

                        placeholder_string = self.vision_bos_token + self.audio_bos_token

                        for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                            if j < len(video_chunk_indexes):
                                video_seq_length = video_chunk_indexes[j][1] - video_chunk_indexes[j][0]
                                placeholder_string += "<|video_placeholder|>" * video_seq_length
                            if j < len(audio_chunk_indexes):
                                audio_seq_length = audio_chunk_indexes[j][1] - audio_chunk_indexes[j][0]
                                placeholder_string += "<|audio_placeholder|>" * audio_seq_length

                        placeholder_string += self.audio_eos_token + self.vision_eos_token
                        sample = sample.replace(
                            self.vision_bos_token + self.video_token + self.vision_eos_token,
                            placeholder_string,
                            1,
                        )

            sample = sample.replace("<|audio_placeholder|>", self.audio_token)
            sample = sample.replace("<|image_placeholder|>", self.image_token)
            sample = sample.replace("<|video_placeholder|>", self.video_token)
            processed_text.append(sample)
        return processed_text

    def get_chunked_index(self, token_indices: np.ndarray, tokens_per_chunk: int) -> list[tuple[int, int]]:
        """
        Splits token index list into chunks based on token value ranges.

        Given a list of token indices, returns a list of (start, end) index tuples representing
        slices of the list where the token values fall within successive ranges of `t_ntoken_per_chunk`.

        For example, if `t_ntoken_per_chunk` is 1000, the function will create chunks such that:
        - the first chunk contains token values < 1000,
        - the second chunk contains values >= 1000 and < 2000, and so on.

        Parameters:
            token_indices (`np.ndarray`): A monotonically increasing list of token index values.
            t_ntoken_per_chunk (`int`): Number of tokens per chunk (used as the chunk size threshold).

        Returns:
            `List[Tuple[int, int]]`: A list of tuples, each representing the start (inclusive)
                                and end (exclusive) indices of a chunk in `token_indices`.
        """
        def _iter():
            i, start_idx = 0, 0  # skip bos token
            current_chunk = 1
            while i < len(token_indices):  # skip eos token
                if token_indices[i] >= current_chunk * tokens_per_chunk:
                    yield (start_idx, i)
                    start_idx = i
                    current_chunk += 1
                i += 1
            yield (start_idx, len(token_indices))
        return list(_iter())


    def compute_audio_token_indices(self, original_audio_length, compress_info, tokens_per_second=25):

        if compress_info is None or len(compress_info) == 0:
            return np.arange(original_audio_length), original_audio_length
        
        num_seconds = original_audio_length // tokens_per_second
        num_seconds = min(num_seconds, len(compress_info))
        

        
        audio_token_indices = []
        
        for sec_idx in range(num_seconds):
            target_tokens = int(compress_info[sec_idx])
            target_tokens = max(1, min(target_tokens, tokens_per_second))
            
            if target_tokens >= tokens_per_second:
                for j in range(tokens_per_second):
                    audio_token_indices.append(sec_idx * tokens_per_second + j)
            else:
                local_indices = np.linspace(0, tokens_per_second - 1, target_tokens, dtype=int)
                for local_idx in local_indices:
                    audio_token_indices.append(sec_idx * tokens_per_second + local_idx)
        processed_tokens = num_seconds * tokens_per_second
        for j in range(processed_tokens, original_audio_length):
            audio_token_indices.append(j)
        
        return np.array(audio_token_indices), len(audio_token_indices)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def apply_chat_template(self, conversations, chat_template=None, **kwargs):
        if isinstance(conversations[0], dict):
            conversations = [conversations]
        for conversation in conversations:
            if (
                conversation[0]["role"] != "system"
                or conversation[0]["content"][0]["text"]
                != "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
            ):
                logging.warning(
                    "System prompt modified, audio output may not work as expected. "
                    + "Audio output mode only works when using default system prompt 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'"
                )
        return super().apply_chat_template(conversations, chat_template, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names
                + feature_extractor_input_names
                + image_processor_input_names
                + ["feature_attention_mask"]
                + ["video_second_per_grid"]
            )
        )


__all__ = ["Qwen2_5OmniProcessor"]