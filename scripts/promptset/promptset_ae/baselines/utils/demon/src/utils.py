import torch
import torchvision.transforms as transforms
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from tqdm import tqdm, trange

from src.config import FILE_PATH, DTYPE, DEVICE, IMAGE_DIMENSION

# Load tokenizers and text encoders.
tokenizers = [
    CLIPTokenizer.from_pretrained(FILE_PATH, subfolder="tokenizer"),
    CLIPTokenizer.from_pretrained(FILE_PATH, subfolder="tokenizer_2")
]
text_encoders = [
    CLIPTextModel.from_pretrained(FILE_PATH, subfolder="text_encoder").to(device=DEVICE, dtype=DTYPE),
    CLIPTextModelWithProjection.from_pretrained(FILE_PATH, subfolder="text_encoder_2").to(device=DEVICE, dtype=DTYPE)
]

# Load the VAE model.
vae = AutoencoderKL.from_pretrained(FILE_PATH, subfolder='vae').to(device=DEVICE, dtype=torch.float32)

# Define transforms.
pil_convert = transforms.ToPILImage()
tensor_convert = transforms.ToTensor()


@torch.inference_mode()
def from_pil_to_latent(pil_img):
    """Convert a PIL image to a latent representation."""
    pil_img = pil_img.resize((IMAGE_DIMENSION, IMAGE_DIMENSION))
    img_tensor = tensor_convert(pil_img).unsqueeze(0).to(device=DEVICE, dtype=DTYPE)
    return encode(img_tensor)


@torch.inference_mode()
def from_latent_to_pil(latents):
    """
    Decode latents into images and convert them to PIL format.
    
    Returns a single PIL image if the batch size is 1; otherwise, a list.
    """
    decoded = decode(latents)
    images = [pil_convert(img) for img in decoded]
    return images[0] if len(images) == 1 else images


@torch.inference_mode()
def get_guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float32):
    """
    Compute guidance scale embeddings (sinusoidal features).

    See: https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
    """
    assert len(w.shape) == 1, "w must be a 1D tensor"
    w = w * 1000.0

    half_dim = embedding_dim // 2
    # Precompute constant factors on the same device as w.
    factor = torch.log(torch.tensor(10000.0, device=w.device)) / (half_dim - 1)
    exp_term = torch.exp(torch.arange(half_dim, device=w.device, dtype=dtype) * -factor)
    emb = w.to(dtype)[:, None] * exp_term[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad if needed
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb


@torch.inference_mode()
def get_condition(text, time_cond=False):
    """
    Generate a condition dictionary from text.
    
    Uses two tokenizers and encoders, concatenates their outputs, and adds auxiliary
    conditioning information. If time_cond is True, includes timestep conditioning.
    """
    # Duplicate the prompt for both encoders.
    prompts = [text, text]
    prompt_embeds_list = []
    pooled_prompt_embeds = None

    for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(DEVICE)
        outputs = text_encoder(text_input_ids, output_hidden_states=True)
        pooled_prompt_embeds = outputs[0]
        # Use the second-to-last hidden state.
        prompt_embeds_list.append(outputs.hidden_states[-2])

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1).to(dtype=text_encoders[1].dtype, device=DEVICE)
    bs, seq_len, _ = prompt_embeds.shape
    # Reshape for consistency.
    prompt_embeds = prompt_embeds.view(bs, seq_len, -1)

    # Auxiliary condition: constant time IDs.
    add_time_ids = torch.tensor([[1024., 1024., 0., 0., 1024., 1024.]],
                                dtype=torch.float16, device=DEVICE).repeat(bs, 1)

    if time_cond:
        guidance_scale_tensor = torch.zeros(bs, device=DEVICE)
        timestep_cond = get_guidance_scale_embedding(guidance_scale_tensor, embedding_dim=256).to(device=DEVICE, dtype=DTYPE)
    else:
        timestep_cond = None

    return {
        "encoder_hidden_states": prompt_embeds,
        "timestep_cond": timestep_cond,
        "added_cond_kwargs": {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids},
    }


@torch.inference_mode()
def encode(input_img):
    """
    Encode an image (in [-1, 1] range) into latent space.
    
    The input is adjusted to the VAE's expected format.
    """
    if input_img.ndim < 4:
        input_img = input_img.unsqueeze(0)
    input_img = input_img.to(device=DEVICE, dtype=torch.float32)
    latent = vae.encode(input_img * 2 - 1)
    return (vae.config.scaling_factor * latent.latent_dist.sample()).to(dtype=DTYPE)


@torch.inference_mode()
def decode(latents):
    """
    Decode latent representations into images.

    Processes the latent batch in small chunks.
    """
    latents = (1 / vae.config.scaling_factor) * latents
    MAX_CHUNK_SIZE = 1
    images = []
    for i in range(0, len(latents), MAX_CHUNK_SIZE):
        chunk = latents[i:i + MAX_CHUNK_SIZE].to(torch.float32)
        decoded_chunk = vae.decode(chunk).sample
        # Map output from [-1, 1] to [0, 1] and clamp.
        decoded_chunk = ((decoded_chunk / 2 + 0.5).clamp(0, 1)).to(latents.dtype)
        images.append(decoded_chunk)
    image = torch.cat(images).to(latents.dtype)
    return image
