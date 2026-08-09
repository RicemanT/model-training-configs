
# Telescopa Full Matrix LOKR Training Diary

Anima-Telescopa is a full matrix LoKr (full finetune substitute) of the [Anima Base v1.0 model by CircleStone Labs](https://huggingface.co/circlestone-labs/Anima/blob/main/split_files/diffusion_models/anima-base-v1.0.safetensors)

The aim of this finetune include:

- Practicing for the first time on how to finetune a image model, particularly of datasets with 10k+ samples
- Attempting to improve Anima background generation quality and ability through the use of a pure manually curated (by hand) aesthetic dataset. (Designated as non important side goal, not likely to work)
- Testing of differences in training practice and results between Full-Finetuning and Full Matrix LoKr, which is a cheaper (but slower) method of training with similar result as regular finetuning [LoRA variant introduced by Kblueleaf and his LyCoRIS library](https://github.com/kohakublueleaf/lycoris) [Arxiv](https://arxiv.org/abs/2309.14859)


## Preamble / Disclaimers

The purpose of this article is for transparency in sharing an up-to-date overview of my training process at the time of this writing, it is not intended to be a guide or indicative of best practices. It's not a professional tech report.

For reflection and error checking purposes, this article is not written with the assistance of a LLM. (I only copied the format from [Motimalu diary of his Anima tune](https://github.com/motimalu/diffusion-training-configs/blob/main/diffusion-pipe/anima/notes/kirazuri4.0-notes%20.md)

The Anima-Telescopa model is produced in an individual hobbyist capacity with no external funding, and the model weights remain open with no additional restrictions to the base model license.

## Training Details Summary

**Trainer:** [Bluvoll diffusion-pipe fork, originally a training program by Anima creator-Tdrussel](https://github.com/bluvoll/diffusion-pipe)

**Training device:** x2  NVIDIA RTX™ A4000 16GB VRAM provided by Astromahdi [gpu-garden](https://gpu.garden/)

**Total training time:** ~3 days or ~68 hours

**Total samples seen(unbatched steps):** ~100,0000 samples

**Training resolutions:**

- 1024^2
- 1280^2

### Training run ( initial 3 epoch run into a further 7 epoch 2nd run)

See: [TelescopaLOKR training config](https://github.com/RicemanT/model-training-configs/blob/main/diffusion-pipe/Anima/configs/TelescopaLOKR.toml)


- **Samples seen(unbatched steps):** ~30,000 + ~70,000 samples
- **Learning Rate:** 3e-6
- **Learning Rate Scheduler:** Constant with Warmups
- **LLM Adaptor Learning Rate:** Disabled
- **Precision:** Full BF16
- **Optimizer:** AdamW8bit with Kahan Summation, utilizing Kahan summation to prevent precision roundoff errors during pure BF16 weight
- **Weight Decay:** 0.01
- **Timestep Sampling Strategy:** Logit-Normal with Shift 4 and Sigmoid Scale=1.3

### Additional Features

- Tag Dropout: 10%
- Caption Dropout: 5%
- Mixed caption with ratio of 25/25/25/25
- Tag Shuffle
- Enable Full Matrix LoKR with:
dim/rank = 16
alpha = 16
dtype = 'bfloat16'
dropout = 0.0
factor = 2 (~800mb file size)


## Dataset

Collected through various online sources, mainly danbooru and [Akanyan personal collection of anime screencap, scattered throughout his account](https://www.reddit.com/r/anime/comments/4lhd5l/complete_monogatari_series_background_art/)

See: The utils folder of this repo, where various dataset tools were stored.

## Dataset Curation and Processing

I first scrapped through a list of artists with on average decent to great background quality on Danbooru and their accompanying tags sidecar, then I collected through Akanyan reddit collection one by one with any series I deemed to have great background arts. In total, the uncurated danbooru data would be roughly around 43k images, and the anime screencap portion would be another 7k-8k images, combining into 50k+ raw images that I have to go through BY HAND.
Yes I'm a nutjob.
It actually wasn't that bad all things considered, you can seriously judge image quality with quick eye scan very quickly (200 images/minute kinda deal). The rough time to curate it all was about two weeks, but I could have finished it in 4-5 days if I had locked in. The finished dataset had 10143 images.

After curation, I processed the dataset by converting all pngs to lossless webp, and downscaling images with absurd resolution down to 2000x pixel limit using cv2's inter_area for efficient training. It has also been cleanly deduped and wiped of images with wide/weird resolution ratio.

There was a watermark/logo removal phase being planned using a YOLO detection -> Iopaint workflow but after some thinking i've decided agaisnt, due to time constraint and current open source detection model not being reliable enough to nail the job for every image consistently.

See: [Dataset Processing notebook](https://github.com/RicemanT/model-training-configs/blob/main/diffusion-pipe/Anima/utils/Dataset_webp_conversion.ipynb)


## Dataset Tagging

See: [Tagging Notebook script](https://github.com/RicemanT/model-training-configs/blob/main/diffusion-pipe/Anima/utils/animetimm-convnextv2huge-tagger.ipynb)

I tagged the dataset with the current SOTA danboour tagger [convnextv2_huge.dbv4-full](https://huggingface.co/animetimm/convnextv2_huge.dbv4-full) by animetimm/DeepGHS on the free Google Colab T4 GPU due to the x4 A4000 gpu server not being available at the moment.
Inspect the notebook script to see the exact tagging strategy.

## Dataset Captioning
See: [VLLM captioning script for use on LightningAI H100 compute](https://github.com/RicemanT/model-training-configs/blob/main/diffusion-pipe/Anima/utils/vllm-captioning.py) and [Captioning system prompt](https://github.com/RicemanT/model-training-configs/blob/main/diffusion-pipe/Anima/utils/Anima-Telescopa-captioning-sysprompt.md)

I choose [Qwen 3.6 27B fp8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8) as the captioning model due to its quality and status alongside Gemma 4 31B as the two best captioning model in local open source. VLLM was the easy program of choice for large scale batch captioning job on the H100. The captioning run took about 3h in total, but was only able to finish ~9k images in non thinking mode, so I finished the last 1k images using Qwen 3.6 27B Q6_K on llama.cpp with x2 A4000.

## Failed Finetune Run

I also attempted a full finetune run with the same settings on the whole x4 A4000, which finished training in 1.5 days. The run itself was successful and the experience was valuable but the model, while stable, turn out to have forgotten much character and artist style knowledge, rendering it a completely failure as a competitive Anima tune. There are reasons as to why it turn out like that, but to be honest I'm a bit clueless because general lr settings between a full matrix lokr and a full finetune should be similar from what I know and seen. My best guess is simply that full matrix lokr train everything, just like full finetuning, but it does not go as deep. Anima has shown to be very sensitive to learning new information compared to past and future models, and 3e-6 is probably still a bit too high for full finetuning with the small batch size i was able to use. The training config can be seen in configs folder if anyone want to take a look regardless.



# Results and future considerations

## Training result

The full matrix LoKR tune went alot better than expected. Compared to recent competitive finetune like Tdrussel Aes B, Motimalu KirazuriV4, duongve AnimaYume and Silvermoong mixes, it probably perform on par if not somewhat worse on certain factors like finegrained details/stability/knowledge retention/artist style accuracy retention. But it has also shown in my general usage to be adept and perhaps slightly better at background comprehension/composition/details compared to the base Anima and other comtemporary finetune.
So for being my first finetune/larger scale training attempt ever, I'm fairly satisfied with the result.



## Considerations

-Make sure to always have a copy of the dataset before deleting permanently anything. This is the biggest rookie mistake there is and I'm still salty about it, but during data transfering from my local laptop to my huggingface dataset repo, I ran out of storage space to continue the job. I assume that the current files that were uploaded was already committed to Huggingface, so I deleted the whole anime screencap section from my laptop, which was a whole 4k images with much of it being curated Monogatari series. The upload didn't commit for some reason, I lost all of the anime screencap data. The Kizumonogatari folder was the only one that somehow survived. The mental anguish of erasing almost 1 week of my work almost made me abandon the project. But I'm already too deep in to quit, so I rescrapped the anime screencap dataset and diversify the series selection. This turn out to be a blessing in disguise, since my old screencap subset was a bit too biased on just Monogatari, which would have negatively impact the model.

-Jpeg have lower file size than webp, do not fucking convert jpegs to webp you idiot. Yes I really did do that. This is what being a poor 3rd world student who didnt have computer access or tech experience does to you.
