"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from tempfile import NamedTemporaryFile
import argparse
import torch
import torchaudio
import gradio as gr
import os
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

from share_btn import community_icon_html, loading_icon_html, share_js, css

MODEL = None
IS_SHARED_SPACE = "radames/MusicGen-Continuation" in os.environ.get("SPACE_ID", "")


def load_model(version):
    print("Loading model", version)
    return MusicGen.get_pretrained(version)


def predict(
    text,
    melody_input,
    duration=30,
    continuation=False,
    continuation_start=0,
    continuation_end=30,
    topk=250,
    topp=0,
    temperature=1,
    cfg_coef=3,
):
    global MODEL
    topk = int(topk)
    if MODEL is None:
        MODEL = load_model("melody")

    if duration > MODEL.lm.cfg.dataset.segment_duration:
        raise gr.Error("MusicGen currently supports durations of up to 30 seconds!")
    if continuation and continuation_end < continuation_start:
        raise gr.Error("The end time must be greater than the start time!")
    MODEL.set_generation_params(
        use_sampling=True,
        top_k=topk,
        top_p=topp,
        temperature=temperature,
        cfg_coef=cfg_coef,
        duration=duration,
    )

    if melody_input:
        melody, sr = torchaudio.load(melody_input)
        melody_duration = melody.shape[-1] / sr
        if melody_duration < duration:
            raise gr.Error("The duration must be greater than the melody duration!")
        # sr, melody = melody_input[0], torch.from_numpy(melody_input[1]).to(MODEL.device).float().t().unsqueeze(0)
        if melody.dim() == 2:
            melody = melody[None]
        if continuation:
            print("\nGenerating continuation\n")
            melody_wavform = melody[
                ..., int(sr * continuation_start) : int(sr * continuation_end)
            ]
            output = MODEL.generate_continuation(
                prompt=melody_wavform,
                prompt_sample_rate=sr,
                descriptions=[text],
                progress=True,
            )
        else:
            print("\nGenerating with melody\n")
            melody_wavform = melody[
                ..., : int(sr * MODEL.lm.cfg.dataset.segment_duration)
            ]
            output = MODEL.generate_with_chroma(
                descriptions=[text],
                melody_wavs=melody_wavform,
                melody_sample_rate=sr,
                progress=True,
            )
    else:
        print("\nGenerating without melody\n")
        output = MODEL.generate(descriptions=[text], progress=False)

    output = output.detach().cpu().float()[0]
    with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
        audio_write(
            file.name,
            output,
            MODEL.sample_rate,
            strategy="loudness",
            loudness_headroom_db=16,
            loudness_compressor=True,
            add_suffix=False,
        )
        waveform_video = gr.make_waveform(file.name)

    return (
        waveform_video,
        (sr, melody_wavform.numpy()) if melody_input else None,
    )


def ui(**kwargs):
    def toggle(choice):
        if choice == "mic":
            return gr.update(source="microphone", value=None, label="Microphone")
        else:
            return gr.update(source="upload", value=None, label="File")

    def check_melody_length(melody_input):
        if not melody_input:
            return gr.update(maximum=0, value=0), gr.update(maximum=0, value=0)
        melody, sr = torchaudio.load(melody_input)
        audio_length = melody.shape[-1] / sr
        if melody.dim() == 2:
            melody = melody[None]
        return gr.update(maximum=audio_length, value=0), gr.update(
            maximum=audio_length, value=audio_length
        )

    def preview_melody_cut(melody_input, continuation_start, continuation_end):
        if not melody_input:
            return gr.update(maximum=0, value=0), gr.update(maximum=0, value=0)
        melody, sr = torchaudio.load(melody_input)
        audio_length = melody.shape[-1] / sr
        if melody.dim() == 2:
            melody = melody[None]

        if continuation_end < continuation_start:
            raise gr.Error("The end time must be greater than the start time!")
        if continuation_start < 0 or continuation_end > audio_length:
            raise gr.Error("The continuation settings must be within the audio length!")
        print("cutting", int(sr * continuation_start), int(sr * continuation_end))
        prompt_waveform = melody[
            ..., int(sr * continuation_start) : int(sr * continuation_end)
        ]

        return (sr, prompt_waveform.numpy())

    with gr.Blocks(css=css) as interface:
        gr.Markdown(
            """
            # MusicGen Continuation
            This is your private demo for [MusicGen](https://github.com/facebookresearch/audiocraft), a simple and controllable model for music generation
            presented at: ["Simple and Controllable Music Generation"](https://huggingface.co/papers/2306.05284)
            """
        )
        if IS_SHARED_SPACE:
            gr.Markdown(
                """
                ⚠ This Space doesn't work in this shared UI ⚠

                <a href="https://huggingface.co/spaces/radames/MusicGen-Continuation?duplicate=true" style="display: inline-block;margin-top: .5em;margin-right: .25em;" target="_blank">
                <img style="margin-bottom: 0em;display: inline;margin-top: -.25em;" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
                to use it privately, or use the <a href="https://huggingface.co/spaces/facebook/MusicGen">public demo</a>
                """
            )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Text(
                        label="Describe your music",
                        lines=2,
                        interactive=True,
                        elem_id="text-input",
                    )
                    with gr.Column():
                        radio = gr.Radio(
                            ["file", "mic"],
                            value="file",
                            label="Melody Condition (optional) File or Mic",
                        )
                        melody = gr.Audio(
                            source="upload",
                            type="filepath",
                            label="File",
                            interactive=True,
                            elem_id="melody-input",
                        )
                with gr.Row():
                    submit = gr.Button("Submit")
                # with gr.Row():
                #     model = gr.Radio(
                #         ["melody", "medium", "small", "large"],
                #         label="Model",
                #         value="melody",
                #         interactive=True,
                #     )
                with gr.Row():
                    duration = gr.Slider(
                        minimum=1,
                        maximum=30,
                        value=10,
                        label="Total Duration",
                        interactive=True,
                    )
                with gr.Row():
                    continuation = gr.Checkbox(value=False, label="Enable Continuation")
                with gr.Row():
                    continuation_start = gr.Slider(
                        minimum=0,
                        maximum=30,
                        step=0.01,
                        value=0,
                        label="melody cut start",
                        interactive=True,
                    )
                    continuation_end = gr.Slider(
                        minimum=0,
                        maximum=30,
                        step=0.01,
                        value=0,
                        label="melody cut end",
                        interactive=True,
                    )
                    cut_btn = gr.Button("Cut Melody").style(full_width=False)
                with gr.Row():
                    preview_cut = gr.Audio(
                        type="numpy",
                        label="Cut Preview",
                    )
                with gr.Accordion(label="Advanced Settings", open=False):
                    with gr.Row():
                        topk = gr.Number(label="Top-k", value=250, interactive=True)
                        topp = gr.Number(label="Top-p", value=0, interactive=True)
                        temperature = gr.Number(
                            label="Temperature", value=1.0, interactive=True
                        )
                        cfg_coef = gr.Number(
                            label="Classifier Free Guidance",
                            value=3.0,
                            interactive=True,
                        )
            with gr.Column():
                output = gr.Video(label="Generated Music", elem_id="generated-video")
                output_melody = gr.Audio(label="Melody ", elem_id="melody-output")
                with gr.Row(visible=False) as share_row:
                    with gr.Group(elem_id="share-btn-container"):
                        community_icon = gr.HTML(community_icon_html)
                        loading_icon = gr.HTML(loading_icon_html)
                        share_button = gr.Button(
                            "Share to community", elem_id="share-btn"
                        )
                        share_button.click(None, [], [], _js=share_js)
        melody.change(
            check_melody_length,
            melody,
            [continuation_start, continuation_end],
            queue=False,
        )
        cut_btn.click(
            preview_melody_cut,
            [melody, continuation_start, continuation_end],
            preview_cut,
            queue=False,
        )

        submit.click(
            lambda x: gr.update(visible=False),
            None,
            [share_row],
            queue=False,
            show_progress=False,
        ).then(
            predict,
            inputs=[
                text,
                melody,
                duration,
                continuation,
                continuation_start,
                continuation_end,
                topk,
                topp,
                temperature,
                cfg_coef,
            ],
            outputs=[output, output_melody],
        ).then(
            lambda x: gr.update(visible=True),
            None,
            [share_row],
            queue=False,
            show_progress=False,
        )
        radio.change(toggle, radio, [melody], queue=False, show_progress=False)
        examples = gr.Examples(
            fn=predict,
            examples=[
                [
                    "An 80s driving pop song with heavy drums and synth pads in the background",
                    "./assets/bach.mp3",
                ],
                [
                    "A cheerful country song with acoustic guitars",
                    "./assets/bolero_ravel.mp3",
                ],
                ["90s rock song with electric guitar and heavy drums", None],
                [
                    "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions",
                    "./assets/bach.mp3",
                ],
                [
                    "lofi slow bpm electro chill with organic samples",
                    None,
                ],
            ],
            inputs=[text, melody],
            outputs=[output],
        )
        gr.Markdown(
            """
            ### More details

            The model will generate a short music extract based on the description you provided.
            You can generate up to 30 seconds of audio.

            We present 4 model variations:
            1. Melody -- a music generation model capable of generating music condition on text and melody inputs. **Note**, you can also use text only.
            2. Small -- a 300M transformer decoder conditioned on text only.
            3. Medium -- a 1.5B transformer decoder conditioned on text only.
            4. Large -- a 3.3B transformer decoder conditioned on text only (might OOM for the longest sequences.)

            When using `melody`, ou can optionaly provide a reference audio from
            which a broad melody will be extracted. The model will then try to follow both the description and melody provided.

            You can also use your own GPU or a Google Colab by following the instructions on our repo.
            See [github.com/facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
            for more details.
            """
        )

        # Show the interface
        launch_kwargs = {}
        username = kwargs.get("username")
        password = kwargs.get("password")
        server_port = kwargs.get("server_port", 0)
        inbrowser = kwargs.get("inbrowser", False)
        share = kwargs.get("share", False)
        server_name = kwargs.get("listen")

        launch_kwargs["server_name"] = server_name

        if username and password:
            launch_kwargs["auth"] = (username, password)
        if server_port > 0:
            launch_kwargs["server_port"] = server_port
        if inbrowser:
            launch_kwargs["inbrowser"] = inbrowser
        if share:
            launch_kwargs["share"] = share

        interface.queue().launch(**launch_kwargs, max_threads=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--listen",
        type=str,
        default="0.0.0.0",
        help="IP to listen on for connections to Gradio",
    )
    parser.add_argument(
        "--username", type=str, default="", help="Username for authentication"
    )
    parser.add_argument(
        "--password", type=str, default="", help="Password for authentication"
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=7860,
        help="Port to run the server listener on",
    )
    parser.add_argument("--inbrowser", action="store_true", help="Open in browser")
    parser.add_argument("--share", action="store_true", help="Share the gradio UI")

    args = parser.parse_args()

    ui(
        username=args.username,
        password=args.password,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        share=args.share,
        listen=args.listen,
    )
