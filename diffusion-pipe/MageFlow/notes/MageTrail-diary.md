
# MageTrail Full-Finetune Training Diary

***MageTrail*** is a Danbooru/E621 proof of concept Full-Finetune of [Microsoft's MageFlow 4B](https://huggingface.co/mage-flow-community/Mage-Flow), using a diversity maximized condensed 41k images dataset as a way to tune booru concept and tags based prompting + Illustration capabilities into the model without havint to tune with the full booru dataset.

My second attempt at fine-tuning an image model on larger scale, this finetune aim to prove to the open source community on MageFlow 4B having good potential as an architecture for further finetuning.
~ While V0.1 is still  very obviously undertrained and unstable, the model has shown great promise in quickly learning and adapting booru concept and tags to its knowledge base
~ The architecture behind MageFlow 4B shows good promise for further investment:
* Being 15-20% faster than NVIDIA Cosmos2/Anima on inference despite being 2 billion parameters larger
* Using MageVAE which perform better than QwenVAE on all usage, only behind the strongest open source VAE currently being Flux2VAE (which it was distilled from), also having the ability to slot in Flux2VAE for inference
* Having a decent Qwen 3 VL 4B Text Encoder
* 256-2048 resolution native support
* Being fairly quick to learn and adapt to new knowledge without any knowledge forgetting


## Preamble / Disclaimers

The purpose of this article is for transparency in sharing an up-to-date overview of my training process at the time of this writing, it is not intended to be a guide or indicative of best practices. It's not a professional tech report.

For reflection and error checking purposes, this article is not written with the assistance of a LLM. (I only copied the format from [Motimalu diary of his Anima tune](https://github.com/motimalu/diffusion-training-configs/blob/main/diffusion-pipe/anima/notes/kirazuri4.0-notes%20.md)

The Anima-Telescopa model is produced in an individual hobbyist capacity with no external funding, and the model weights remain open with no additional restrictions to the base model license.

## Training Details Summary

|                                          |                                                                                                                    |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Base model**                           | [Microsoft's MageFlow 4B](https://huggingface.co/mage-flow-community/Mage-Flow)                                    |
| **Method**                               | Full Finetune                                                                                                      |
| **Trainer**                              | [My Diffusion-Pipe fork](https://github.com/RicemanT/diffusion-pipe-mageflow-ft)                                   |
| **Hardware**                             | x8 H100 HBM3 80GB, courtesy of Banodoco grant                                                                      |
| **Total training time**                  | 8 hours (~64 H100 hours)                                                                                           |
| **Total samples seen**                   | ~ 326656                                                                                                           |
| **Training resolutions**                 | 1024²     

### Training run 

### Version 0.1 (initial 20 epoch run → extended 10 epoch run)

**Budget:** 130~ dollars (25-30 lost due to experiments and mistakes)

Full config: [Training](https://github.com/RicemanT/model-training-configs/blob/main/diffusion-pipe/MageFlow/configs/V0.1/mage_flow_BooruEssenceFFT.toml) and [Dataset](https://github.com/RicemanT/model-training-configs/blob/main/diffusion-pipe/MageFlow/configs/V0.1/mage_flow_BooruEssenceDataset.toml)

- **Learning rate:** 7e-6
- **LR scheduler:** Warmup -> Constant -> REX to 0e-7 
- **Precision:** Full BF16
- **Optimizer:** AdamW8bit with Kahan summation (to offset BF16 precision roundoff)
- **Weight decay:** 0.02
- **Timestep sampling:** Logit-Normal, shift 6, sigmoid scale 1.0


### Additional training features

- Tag dropout: 10%
- Caption dropout: 5%
- Mixed captions at 25/25/25/25 ratio (tags only, NL only, tags-nl, nl-tags)
- Tag shuffle
- Caption shuffle
- Artist trigger attribution system

The artist trigger attribution is a feature that I added to my trainer fork and it basically does this:
*Since my dataset tags and caption sidecar all begin with the [Drawn by artistname] trigger format, the resulting combined tags-nl and nl-tags variant became "Drawn by artistname, 1girl, black hair, pink jacket, Drawn by artistname. The girl with the black hair and pink jacket....", not ideal right? The artist trigger appears twice, which mean if I train like that then users have to prompt the trigger twice to activate the artist style, blergh.
-> The new system I created just basically dedupe the trigger to always have only one in the tags-nl and nl-tags combination, if the tags section went first then it get to keep its artist trigger while the nl section behind get deduped.


## Dataset

The original version of this dataset is [Lodestone-Rock Booru-Essence](https://huggingface.co/datasets/lodestones/booru-essence), but it used 2024 tag definition and is captioned with old outdated captioners.

So I went ahead and update tag definition to July/August 2026 and most importantly:

-Tagged additional tags using the https://huggingface.co/animetimm/convnextv2_huge.dbv4-full model for Danbooru, and https://huggingface.co/RedRocket/Hydra for E621.

-Recaptioned the whole thing with the current best vision captioner available **Gemini 3.7 Flash** while using **Grok 4.6** and **Qwen 3.8 27B** as good backups for NSFW refusals.

Unlike the original dataset, the artist trigger tag format was changed from **by artistname** to **Drawn by artistname**. This is my personal preference for my own training projects, you can use scripts to change the prepend format to your liking.

The cost of this whole operation on my own pocket was around 70 dollars, but actual usage if considering the free Vertex credits would be 400-450 dollars ;-;


## Dataset Tagging

See: [Tagging Notebook](https://github.com/RicemanT/model-training-configs/blob/main/diffusion-pipe/MageFlow/utils/modal_tagger.ipynb)

Inspect the notebook script to see the exact tagging strategy.

## Dataset Captioning
See: 

[VLLM captioning script for Qwen 3.8 27B](https://github.com/RicemanT/model-training-configs/blob/main/diffusion-pipe/MageFlow/utils/vllm.py)

[Vertex captioning script for Gemini 3.7 Flash](https://github.com/RicemanT/model-training-configs/blob/main/diffusion-pipe/MageFlow/utils/vertex-captioning.py) 

[LinkAPI captioning script for Grok 4.6](https://github.com/RicemanT/model-training-configs/blob/main/diffusion-pipe/MageFlow/utils/linkapi-caption.py)

and [Captioning system prompt](https://github.com/RicemanT/model-training-configs/blob/main/diffusion-pipe/MageFlow/utils/Booru-Essence-2026-sysprompt.md)

I choose Gemini 3.7 Flash because it was the best vision model available in the world at the time of processing, while Grok 4.6 perform solidly as a more lenient api option with capable vision, and Qwen 3.8 as capable local option with adequate vision.

# Results and future considerations

## Training result

## Version 0.1
It went fantastic actually, honestly. I was very surprised that MageFlow learn that well on such midget budget, nothing more to say really, I still see room for it to converge and stabilize for actual usage.





## Considerations

-Be careful with mounted drive, writable disk storage and non writable disk storage, fumbling around on this issue cost me 20 dollars due to wasting time on the x8 H100 pod

