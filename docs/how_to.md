# Stable-Diffusion-FT

## **Stable diffusion Fine-Tuning with LoRA**

### **LoRA**

pretrained 모델의 weights를 고정하고, 그 위에 새로운 task를 위한 weights를 추가하는 방식으로 fine tuning을 진행. 더 적은 양의 데이터로 fine tuning 가능함


## 환경 구축
```git clone https://github.com/huggingface/diffusers```

accelerate
```accelerate config```
```
----------------------------------------------------------------------------------------------------------------------------
In which compute environment are you running?
This machine
----------------------------------------------------------------------------------------------------------------------------
Which type of machine are you using?
No distributed training

Do you want to run your training on CPU only (even if a GPU is available)? [yes/NO]:
no

Do you wish to optimize your script with torch dynamo?[yes/NO]:
no

Do you want to use DeepSpeed? [yes/NO]: 
no

What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:
all
----------------------------------------------------------------------------------------------------------------------------
Do you wish to use FP16 or BF16 (mixed precision)?
no > 오류나서 no로 함 아니면 fp16

accelerate configuration saved at /home/wfng/.cache/huggingface/accelerate/defaul
```


## Dataset
https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions

- 실행 후 huggingface token 넣어주면 됨

## Training
```
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="/sddata/finetune/lora/pokemon"
export HUB_MODEL_ID="pokemon-lora"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

accelerate launch  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --push_to_hub \
  --hub_model_id=${HUB_MODEL_ID} \
  --report_to=wandb \
  --checkpointing_steps=1500 \
  --validation_prompt="A pokemon with blue eyes." \
  --seed=1337 \
  --resume_from_checkpoint=/sddata/finetune/lora/pokemon/checkpoint-5000
```
MODEL_NAME: 사용할 사전 훈련된 모델의 이름 또는 경로를 저장

OUTPUT_DIR: 학습 중에 생성된 모델과 결과물이 저장될 디렉토리 경로를 지정

HUB_MODEL_ID: 허브(Hugging Face Hub)에 모델을 업로드할 때 사용할 모델의 고유 식별자를 지정

DATASET_NAME: 사용할 데이터셋의 이름을 지정

---

--dataloader_num_workers: 데이터를 로드하는 데 사용할 worker의 수를 지정함

--resolution: 이미지의 해상도를 설정함

--center_crop: 이미지를 중앙을 기준으로 잘라내는 옵션을 설정함

--random_flip: 이미지를 무작위로 좌우 반전시킴

--train_batch_size: 학습할 때 사용할 배치 크기를 설정함

--gradient_accumulation_steps: 그래디언트를 누적하는 단계를 설정함

--max_train_steps: 학습의 최대 스텝 수를 지정함

--learning_rate: 학습률을 설정함

--max_grad_norm: 그래디언트의 최대 정규화를 결정함

--lr_scheduler: 학습률 스케줄러를 선택함

--lr_warmup_steps: 학습률을 웜업하기 위한 스텝 수를 지정함

--output_dir: 학습 결과물을 저장할 디렉토리 경로를 설정함

--push_to_hub: 학습된 모델을 Hugging Face Hub에 업로드할지 여부를 결정함

--report_to: 학습 결과를 보고할 플랫폼을 선택함 여기서는 Weights & Biases(W&B)를 선택함

--checkpointing_steps: 체크포인트를 저장할 간격을 설정함

--validation_prompt: 검증 시 사용할 텍스트 프롬프트를 지정함

--seed: 랜덤 시드를 설정함

--resume_from_checkpoint: 이전 체크포인트에서 학습을 재개할 때 사용할 체크포인트의 경로를 설정함


- vim으로 script 수정 후 실행
ex) `./script.sh`

![image](https://github.com/harim061/Stable-Diffusion-FT/assets/90364684/2d264d28-424f-4121-a7b2-4a9546cbfda8)

--- 
발생했던 오류 및 이슈
1. `wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results`
- 해결 : `wandb disabled`
  
2. `safetensors_rust.SafetensorError: Error while serializing: IoError(Os { code: 28, kind: StorageFull, message: "No space left on device" })`
- device 메모리 부족 오류.
- 해결 : checkpoint step증가
- 용량 확인 방법 `du -sh 경로`
- 사용하지 않는 checkpoint dir 삭제 `rm -rf 이름`

3. checkpoint로 이어서 학습하기
`  --resume_from_checkpoint=/sddata/finetune/lora/pokemon/checkpoint-5000`


---
## 가중치 사용

```
from diffusers import StableDiffusionPipeline
import torch
device = "cuda"

# load model

model_path = "/sddata/finetune/lora/pokemon/checkpoint-12000/pytorch_lora_weights.safetensors"

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False
)

# load lora weights
pipe.unet.load_attn_procs(model_path)
# set to use GPU for inference
pipe.to(device)

# generate image
prompt = "a drawing of girrafe pokemon"
image = pipe(prompt, num_inference_steps=30).images[0]
# save image
image.save("image.png")

image
```

