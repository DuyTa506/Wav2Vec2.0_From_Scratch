# WAV2VEC 2.0 from scratch

1. [Phase 1 : Pretraining](#Pretraining)
2. [Phase 2 : Finetune](#feature)

<a name = "Important note" ></a>
You only need install the requirements at the base work dir, and don't need to install requirements on each child work dir
```
pip install -r requirements.txt
```
</br>
<a name = "Pretraining" ></a>
### Finetune
Suppose you want to reproduct Wav2vec2.0 model from scratch , firstly, prepare the datasets for pretraining
Change the work dir to [Wav2vec2-Pretraining](Wav2vec2-Pretraining) and check [README](Wav2vec2-Pretraining/README.md) for further detail :D
</br>

<a name = "Finetune" ></a>
### Finetune
Certainly, you can pass phase 1 if you had strong enough pretrained model, check [this](https://huggingface.co/TencentGameMate) out
Yah, and after that, change the work dir to [Wav2vec2-Finetune](Wav2vec2-Finetune) and check [README](Wav2vec2-Finetune/README.md) for further detail !!
</br>
