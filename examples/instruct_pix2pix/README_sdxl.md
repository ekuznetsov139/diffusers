# InstructPix2Pix SDXL training example

***This is based on the original InstructPix2Pix training example.***

[Stable Diffusion XL](https://huggingface.co/papers/2307.01952) (or SDXL) is the latest image generation model that is tailored towards more photorealistic outputs with more detailed imagery and composition compared to previous SD models. It leverages a three times larger UNet backbone. The increase of model parameters is mainly due to more attention blocks and a larger cross-attention context as SDXL uses a second text encoder. 

The `train_instruct_pix2pix_xl.py` script shows how to implement the training procedure and adapt it for Stable Diffusion XL.

***Disclaimer: Even though `train_instruct_pix2pix_xl.py` implements the InstructPix2Pix
training procedure while being faithful to the [original implementation](https://github.com/timothybrooks/instruct-pix2pix) we have only tested it on a [small-scale dataset](https://huggingface.co/datasets/fusing/instructpix2pix-1000-samples). This can impact the end results. For better results, we recommend longer training runs with a larger dataset. [Here](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered) you can find a large dataset for InstructPix2Pix training.***

## Running locally with PyTorch

### Installing the dependencies

Refer to the original InstructPix2Pix training example for installing the dependencies.

You will also need to get access of SDXL by filling the [form](https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9). 

### Toy example

As mentioned before, we'll use a [small toy dataset](https://huggingface.co/datasets/fusing/instructpix2pix-1000-samples) for training. The dataset 
is a smaller version of the [original dataset](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered) used in the InstructPix2Pix paper.

Configure environment variables such as the dataset identifier and the Stable Diffusion
checkpoint:

```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-0.9"
export DATASET_ID="fusing/instructpix2pix-1000-samples"
```

Now, we can launch training:

```bash
python train_instruct_pix2pix_xl.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_ID \
    --enable_xformers_memory_efficient_attention \
    --resolution=256 --random_flip \
    --train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing \
    --max_train_steps=15000 \
    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
    --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --seed=42 
```

Additionally, we support performing validation inference to monitor training progress
with Weights and Biases. You can enable this feature with `report_to="wandb"`:

```bash
python train_instruct_pix2pix_xl.py \
    --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-0.9 \
    --dataset_name=$DATASET_ID \
    --use_ema \
    --enable_xformers_memory_efficient_attention \
    --resolution=512 --random_flip \
    --train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing \
    --max_train_steps=15000 \
    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
    --learning_rate=5e-05 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --seed=42 \
    --val_image_url_or_path="https://datasets-server.huggingface.co/assets/fusing/instructpix2pix-1000-samples/--/fusing--instructpix2pix-1000-samples/train/23/input_image/image.jpg" \
    --validation_prompt="make it in japan" \
    --report_to=wandb
 ```

 We recommend this type of validation as it can be useful for model debugging. Note that you need `wandb` installed to use this. You can install `wandb` by running `pip install wandb`. 

 [Here](https://wandb.ai/sayakpaul/instruct-pix2pix/runs/ctr3kovq), you can find an example training run that includes some validation samples and the training hyperparameters.

 ***Note: In the original paper, the authors observed that even when the model is trained with an image resolution of 256x256, it generalizes well to bigger resolutions such as 512x512. This is likely because of the larger dataset they used during training.***

 ## Training with multiple GPUs

`accelerate` allows for seamless multi-GPU training. Follow the instructions [here](https://huggingface.co/docs/accelerate/basic_tutorials/launch)
for running distributed training with `accelerate`. Here is an example command:

```bash 
accelerate launch --mixed_precision="fp16" --multi_gpu train_instruct_pix2pix.py \
    --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-0.9 \
    --dataset_name=$DATASET_ID \
    --use_ema \
    --enable_xformers_memory_efficient_attention \
    --resolution=512 --random_flip \
    --train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing \
    --max_train_steps=15000 \
    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
    --learning_rate=5e-05 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --seed=42 \
    --val_image_url_or_path="https://datasets-server.huggingface.co/assets/fusing/instructpix2pix-1000-samples/--/fusing--instructpix2pix-1000-samples/train/23/input_image/image.jpg" \
    --validation_prompt="make it in japan" \
    --report_to=wandb
```

 ## Inference

 Once training is complete, we can perform inference:

 ```python
import PIL
import requests
import torch
from diffusers import StableDiffusionXLInstructPix2PixPipeline

model_id = "your_model_id" # <- replace this 
pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
generator = torch.Generator("cuda").manual_seed(0)

url = "https://datasets-server.huggingface.co/assets/fusing/instructpix2pix-1000-samples/--/fusing--instructpix2pix-1000-samples/train/23/input_image/image.jpg"


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

image = download_image(url)
prompt = "make it Japan"
num_inference_steps = 20
image_guidance_scale = 1.5
guidance_scale = 10

edited_image = pipe(prompt, 
    image=image, 
    num_inference_steps=num_inference_steps, 
    image_guidance_scale=image_guidance_scale, 
    guidance_scale=guidance_scale,
    generator=generator,
).images[0]
edited_image.save("edited_image.png")
```

We encourage you to play with the following three parameters to control
speed and quality during performance:

* `num_inference_steps`
* `image_guidance_scale`
* `guidance_scale`

Particularly, `image_guidance_scale` and `guidance_scale` can have a profound impact
on the generated ("edited") image (see [here](https://twitter.com/RisingSayak/status/1628392199196151808?s=20) for an example).

If you're looking for some interesting ways to use the InstructPix2Pix training methodology, we welcome you to check out this blog post: [Instruction-tuning Stable Diffusion with InstructPix2Pix](https://huggingface.co/blog/instruction-tuning-sd). 
