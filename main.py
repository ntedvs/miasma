import os

from dia.model import Dia
from moviepy import AudioFileClip, CompositeVideoClip, VideoFileClip
from praw import Reddit

reddit = Reddit(
    client_id=os.getenv("ID"),
    client_secret=os.getenv("SECRET"),
    user_agent="linux:miasma:0.1.0:czenty",
)

model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16")

submission = next(reddit.subreddit("askreddit").hot(limit=1))
text = [submission.title]

submission.comment_sort = "best"
for comment in submission.comments[:10]:
    text.append(comment.body)

formatted = []

for i, item in enumerate(text):
    prefix = "[S1]" if i % 2 == 0 else "[S2]"

    if i != 0:
        formatted.append(f"{prefix} {i}. {item}")
    else:
        formatted.append(f"{prefix} {item}")

output = model.generate(" ".join(formatted), use_torch_compile=False, verbose=True)
model.save_audio("output.mp3", output)

audio = AudioFileClip("output.mp3")

clip = (
    VideoFileClip("footage.webm")
    .without_audio()
    .with_audio(audio)
    .subclipped(0, audio.duration)
)

height = clip.h
width = int(height * 9 / 16)

x1 = (clip.w - width) // 2
x2 = x1 + width

clip = clip.cropped(x1, 0, x2, 0)
CompositeVideoClip([clip]).write_videofile("output.mp4")
