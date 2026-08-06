# VideoTranscriptor

Pull the audio out of a video and transcribe it two ways:

1. **Deepgram** — the hosted API. Fast, accurate, needs a key and a network.
2. **NVIDIA Canary 1B** — an encoder-decoder speech model run locally through
   NeMo. No key, no network, no per-minute cost; wants a GPU and several GB of
   disk.

Run either on its own, or run both and get a report showing exactly where they
disagree.

```
$ videotranscriptor talk.mp4

=== talk.mp4 ===
  extracting audio from talk.mp4
  audio ready: talk.wav (28.4 MB, 16 kHz mono)
  [deepgram] uploading 28.4 MB to Deepgram (nova-3)
  [deepgram] received 2831 words in 9.4s
  [deepgram] wrote json, srt, txt
  [canary] loading nvidia/canary-1b onto cuda (first run downloads several GB)
  [canary] split 933.2s of audio into 34 chunks (<= 30s each)
  [canary] decoded 2794 words in 71.6s
  [canary] wrote json, srt, txt
  wrote comparison report to talk.comparison.md

  deepgram  2831 words, 210 segments, 0.01x realtime
            json talk_transcripts/talk.deepgram.json
            srt  talk_transcripts/talk.deepgram.srt
            txt  talk_transcripts/talk.deepgram.txt
  canary    2794 words, 34 segments, 0.08x realtime
            json talk_transcripts/talk.canary.json
            srt  talk_transcripts/talk.canary.srt
            txt  talk_transcripts/talk.canary.txt

  deepgram vs canary
    word agreement : 94.1%
    substitutions  : 121
    only in deepgra: 47
    only in canary : 10
    speed          : deepgram 0.01x / canary 0.08x
    report         : talk_transcripts/talk.comparison.md
```

## Install

ffmpeg is required for both approaches — it does the demuxing and decoding.

```bash
brew install ffmpeg            # macOS
sudo apt-get install ffmpeg    # Debian/Ubuntu
winget install Gyan.FFmpeg     # Windows
```

Then the package:

```bash
pip install -e .                  # cloud approach only (small install)
pip install -e ".[local]"         # adds torch + NeMo for Canary
pip install -e ".[local,dev]"     # ...and pytest
```

`[local]` pulls in PyTorch and `nemo_toolkit[asr]`, which is a multi-gigabyte
dependency tree. Skip it if you only want the Deepgram path.

Set your Deepgram key ([console.deepgram.com](https://console.deepgram.com)):

```bash
export DEEPGRAM_API_KEY=...
# or copy .env.example to .env and fill it in — a real environment
# variable always wins over the file
```

Check your setup at any time:

```bash
videotranscriptor doctor      # ffmpeg, credentials, local model deps
videotranscriptor backends    # which approaches are usable right now
```

## Usage

```bash
videotranscriptor talk.mp4                      # both approaches + comparison
videotranscriptor talk.mp4 -b deepgram          # cloud only
videotranscriptor talk.mp4 -b canary            # local only
videotranscriptor talk.mp4 -b auto              # first one that is configured
videotranscriptor *.mp4 -f srt -o subtitles/    # batch, subtitles only
videotranscriptor talk.mp4 -l de                # tell it the language
videotranscriptor podcast.m4a                   # audio files work too
```

Useful flags:

| flag | what it does |
| --- | --- |
| `-b, --backend` | `deepgram`, `canary`, `both` (default), or `auto` |
| `-f, --format` | `txt`, `srt`, `vtt`, `json`, `md` — repeatable, default `txt srt json` |
| `-o, --output-dir` | default is `<input>_transcripts/` next to the input |
| `-l, --language` | ISO code, or `auto` (default) |
| `--diarize` | label speakers (Deepgram only) |
| `--device` | `cuda` or `cpu` for Canary; autodetected otherwise |
| `--chunk-seconds` | Canary's audio window, default 30 |
| `--keep-audio` | keep the extracted 16 kHz WAV |
| `--keep-raw` | include the raw backend response in the JSON output |
| `--no-compare` | skip the comparison report |

Exit codes: `0` everything worked, `2` some backends worked and some did not,
`1` nothing worked.

### Output layout

```
talk_transcripts/
├── talk.deepgram.txt
├── talk.deepgram.srt
├── talk.deepgram.json
├── talk.canary.txt
├── talk.canary.srt
├── talk.canary.json
└── talk.comparison.md
```

## How the two approaches work

Audio is extracted **once** — 16 kHz mono PCM WAV — and handed to both
backends, so they are judged on byte-identical input.

### Deepgram

A single POST to the pre-recorded `/v1/listen` endpoint with `utterances` and
`smart_format` on. The response carries word-level timings, so segments land on
real sentence boundaries and the SRT is properly cued. Transient failures
(429, 5xx) are retried with exponential backoff; a 401 fails immediately, since
retrying a bad key never helps.

### Canary

Canary attends over a bounded audio window — around 40 seconds for
`nvidia/canary-1b` — so longer recordings have to be cut up. Cutting mid-word
costs accuracy, so the chunker runs ffmpeg's `silencedetect` first and pulls
each boundary back to the middle of the last silence inside the window. Only
when a whole 30-second window contains no silence at all does it cut on the
hard limit.

Chunks are decoded as a batch and stitched back together with their offsets
restored. Canary returns no timing *inside* a chunk, so its timestamps are
chunk-granular — good enough to navigate a transcript, coarser than Deepgram's
if you are cutting subtitles.

`canary-1b` officially covers English, German, French and Spanish, and it does
not detect language — `--language` is taken as given and defaults to English.
Point `--canary-model` at another NeMo checkpoint (for example
`nvidia/canary-1b-flash`) if you want a different trade-off.

### Which to use

| | Deepgram | Canary 1B |
| --- | --- | --- |
| setup | API key | ~7 GB download, torch + NeMo |
| runs | on Deepgram's servers | on your machine |
| cost | per minute of audio | electricity |
| speed | ~0.01x realtime | ~0.08x realtime on a modern GPU, near or above 1x on CPU |
| privacy | audio leaves the machine | audio never leaves |
| timestamps | word level | chunk level |
| offline | no | yes, after the first download |
| languages | many, auto-detected | 4 for `canary-1b`, must be specified |

## The comparison report

Neither transcript is ground truth, so the report does not claim an accuracy
score. It measures **agreement**: a word-level Levenshtein alignment between
the two, after normalising away case and punctuation so that "Dr. Smith," and
"dr smith" do not register as a disagreement.

```markdown
| word agreement | 94.1% |
| substitutions | 121 |
| deletions (missing from canary) | 47 |
| insertions (extra in canary) | 10 |

## Largest disagreements (25 of 178)

- word ~412: deepgram: 'Kubernetes'  |  canary: 'coup or netties'
```

High agreement means both models heard the same thing and you can trust the
transcript. The disagreement list is where a human should actually look —
proper nouns, jargon and crosstalk cluster there.

## Python API

```python
from pathlib import Path
from videotranscriptor import backends, pipeline
from videotranscriptor.compare import compare

result = pipeline.run(
    Path("talk.mp4"),
    backends=[backends.build("deepgram"), backends.build("canary", device="cuda")],
    output_formats=["srt", "json"],
)

for run in result.successful:
    print(run.backend, run.transcript.word_count, run.outputs["srt"])

print(result.comparison.to_markdown())
```

Single backend, no file output:

```python
from videotranscriptor.backends import DeepgramBackend
from videotranscriptor.pipeline import prepared_audio

with prepared_audio(Path("talk.mp4")) as wav:
    transcript = DeepgramBackend().transcribe(wav, language="en")

print(transcript.text)
for segment in transcript.segments:
    print(segment.start, segment.end, segment.text)
```

## Adding a backend

Implement `TranscriptionBackend` (`check()`, `transcribe()`, `model_id`) and
register it in `backends/__init__.py`. Everything downstream — the writers, the
comparison, the CLI — works off `Transcript`, so nothing else needs to change.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The suite covers chunk planning, silence parsing, the subtitle writers, the
alignment maths and the pipeline's failure handling. It runs in well under a
second and needs neither ffmpeg, a network, nor a model — backends are stubbed.

## Troubleshooting

**`Missing required binaries: ffmpeg`** — install ffmpeg (see above) and make
sure it is on `PATH`.

**`No audio stream found`** — the file has no audio track. Check with
`ffprobe your.mp4`.

**Canary is very slow** — you are on CPU. `videotranscriptor doctor` says so.
Expect roughly realtime or worse; use `--device cuda` on a GPU box, or fall
back to `-b deepgram`.

**Canary's first run stalls for a long time** — it is downloading the
checkpoint (several GB) into `~/.cache/torch/NeMo`. Later runs load from cache.

**Deepgram returns 401** — the key is wrong or unset. `videotranscriptor
doctor` shows which key it found.

**Out of memory on GPU** — lower `--batch-size` (default 4), then
`--chunk-seconds`.
