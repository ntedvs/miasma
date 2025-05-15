import os

import torch
import whisperx
from dia.model import Dia
from moviepy import (
    AudioFileClip,
    CompositeVideoClip,
    TextClip,
    VideoFileClip,
    concatenate_audioclips,
)
from praw import Reddit


def gen(text):
    return model.generate(text, use_torch_compile=False, verbose=True)


comments = 5
aspect_ratio = 9 / 16
audio_file = "audio.mp3"
out_file = "output.mp4"
device = "cuda"

reddit = Reddit(
    client_id=os.getenv("ID"),
    client_secret=os.getenv("SECRET"),
    user_agent="linux:miasma:0.1.0:czenty",
)

model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16")

submission = next(reddit.subreddit("askreddit").hot(limit=1))
text = [submission.title]
model.save_audio("title.mp3", gen(submission.title))

submission.comment_sort = "best"
for i, comment in enumerate(submission.comments[:comments]):
    text.append(comment.body)
    model.save_audio(f"{i}.mp3", gen(f"{i + 1}. {comment.body}"))

audio = concatenate_audioclips(
    [AudioFileClip("title.mp3")] + [AudioFileClip(f"{i}.mp3") for i in range(comments)],
)
audio.write_audiofile(audio_file)

torch.cuda.empty_cache()

model = whisperx.load_model("large-v2", device, compute_type="float16")
wa = whisperx.load_audio(audio_file)
result = model.transcribe(wa, batch_size=16)

align, metadata = whisperx.load_align_model(language_code="en", device=device)
result = whisperx.align(
    result["segments"], align, metadata, wa, device, return_char_alignments=False
)

torch.cuda.empty_cache()

text_clips = []

for segment in result["segments"]:
    for info in segment.get("words", []):
        start = info["start"]
        end = info["end"]
        word = info["word"]

        text = (
            TextClip(
                font="arial.ttf",
                text=word,
                font_size=100,
                color="white",
                stroke_color="black",
                stroke_width=10,
            )
            .with_position("center")
            .with_start(start)
            .with_end(end)
        )

        text_clips.append(text)

clip = (
    VideoFileClip("footage.mp4")
    .without_audio()
    .with_audio(audio)
    .subclipped(0, audio.duration)
)

height = clip.h
width = height * aspect_ratio
x1 = (clip.w - width) / 2
x2 = x1 + width

clip = clip.cropped(x1, 0, x2, 0)
CompositeVideoClip([clip] + text_clips).write_videofile(out_file)
