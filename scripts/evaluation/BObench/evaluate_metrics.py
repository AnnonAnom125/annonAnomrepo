import argparse
import glob
import json
import os
import utils.vila.coca_vila as coca_vila
import utils.vila.coca_vila_configs as coca_vila_configs
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..","..")))
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
from packaging import version
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm import tqdm
from utils.pic_aes_clip_hpsv2 import clip_utils, aes_utils, hps_utils, pickscore_utils
import ImageReward as RM
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
from absl import app
from absl import flags
from absl import logging
import jax
import clip
import jax.numpy as jnp
from lingvo import compat as tf
from lingvo.core import tokenizers as lingvo_tokenizers
from paxml import checkpoints
from paxml import learners
from paxml import tasks_lib
from paxml import train_states
from praxis import base_layer
from praxis import optimizers
from praxis import pax_fiddle
from praxis import py_utils
from praxis import schedules
from torchvision import transforms
NestedMap = py_utils.NestedMap
from utils.clipi_clipt_dino_lpips.anybench_utils import eval_clip_i, eval_clip_t


_PRE_CROP_SIZE = 272
_IMAGE_SIZE = 224
_MAX_TEXT_LEN = 64
_TEXT_VOCAB_SIZE = 64000
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outpath",
        type=str,
        default=None,
        required=True,
        help="Path to read samples and output scores"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="picscore",
        help="Metric to evaluate [picscore clip hpsv2 aesthetic imagereward topiq musiq liqe psnr lpips ssim clipi dino]"
    )
    parser.add_argument(
        "--aes_model_path",
        type=str,
        default="./models/checkpoints/sac+logos+ava1-l14-linearMSE.pth",
        help="Path to model aes"
    )
    parser.add_argument(
        "--hps_model_path",
        type=str,
        default="./models/checkpoints/HPS_v2_compressed.pt",
        help="Path to model hpsv2"
    )
    parser.add_argument(
        "--vila_model_path",
        type=str,
        default="./models/checkpoints/vila_rank_tuned",
        help="Path to model aes"
    )
    parser.add_argument(
        "--prompt_in_filename",
        type=bool,
        default=False,
        help="is the full prompt in the filename or truncated"
    )
    args = parser.parse_args()
    return args

def load_vila_model(
    ckpt_dir,
):
  """Loads the VILA model from checkpoint directory.

  Args:
    ckpt_dir: The path to checkpoint directory

  Returns:
    VILA model, VILA model states
  """
  coca_config = coca_vila_configs.CocaVilaConfig()
  coca_config.model_type = coca_vila.CoCaVilaRankBasedFinetune
  coca_config.decoding_max_len = _MAX_TEXT_LEN
  coca_config.text_vocab_size = _TEXT_VOCAB_SIZE
  model_p = coca_vila_configs.build_coca_vila_model(coca_config)
  model_p.model_dims = coca_config.model_dims
  model = model_p.Instantiate()

  dummy_batch_size = 4  # For initialization only
  text_shape = (dummy_batch_size, 1, _MAX_TEXT_LEN)
  image_shape = (dummy_batch_size, _IMAGE_SIZE, _IMAGE_SIZE, 3)
  input_specs = NestedMap(
      ids=jax.ShapeDtypeStruct(shape=text_shape, dtype=jnp.int32),
      image=jax.ShapeDtypeStruct(shape=image_shape, dtype=jnp.float32),
      paddings=jax.ShapeDtypeStruct(shape=text_shape, dtype=jnp.float32),
      # For initialization only
      labels=jax.ShapeDtypeStruct(shape=text_shape, dtype=jnp.float32),
      regression_labels=jax.ShapeDtypeStruct(
          shape=(dummy_batch_size, 10), dtype=jnp.float32
      ),
  )
  prng_key = jax.random.PRNGKey(123)
  prng_key, _ = jax.random.split(prng_key)
  vars_weight_params = model.abstract_init_with_metadata(input_specs)

  # `learner` is only used for initialization.
  learner_p = pax_fiddle.Config(learners.Learner)
  learner_p.name = 'learner'
  learner_p.optimizer = pax_fiddle.Config(
      optimizers.ShardedAdafactor,
      decay_method='adam',
      lr_schedule=pax_fiddle.Config(schedules.Constant),
  )
  learner = learner_p.Instantiate()

  train_state_global_shapes = tasks_lib.create_state_unpadded_shapes(
      vars_weight_params, discard_opt_states=False, learners=[learner]
  )
  model_states = checkpoints.restore_checkpoint(
      train_state_global_shapes, ckpt_dir
  )
  return model, model_states


def preprocess_image(
    image_path, pre_crop_size, image_size
):
  """Image preprocessing."""
  with tf.compat.v1.gfile.FastGFile(image_path, 'rb') as f:
    image_bytes = f.read()
  image = tf.io.decode_image(image_bytes, 3, expand_animations=False)
  image = tf.image.resize_bilinear(
      tf.expand_dims(image, 0), [pre_crop_size, pre_crop_size]
  )
  image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
  image = tf.cast(image, tf.float32)
  image = image / 255.0
  image = tf.clip_by_value(image, 0.0, 1.0)
  return image.numpy()

def main(args):

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outpath=args.outpath
    if args.metric == "clip":
        model = clip_utils.Selector(device)
    elif args.metric == "aesthetic":
        model = aes_utils.Selector(device, args.aes_model_path)
    elif args.metric == "hpsv2":
        model = hps_utils.Selector(device, args.hps_model_path)
    elif args.metric == "picscore":
        model = pickscore_utils.Selector(device)
    elif args.metric == "imagereward":
        model = RM.load("ImageReward-v1.0")
        model = model.to(device)
    elif args.metric in ["topiq","musiq","liqe","psnr","lpips","ssim"]:
        from utils.topiq_musiq_liqe_psnr_ssim_lpips import metric_sel
        model = metric_sel.Selector(args.metric, device)
    elif args.metric == "clipi":
        clip_model, transform = clip.load("ViT-B/32")
    elif args.metric == "vila":
        model, model_states = load_vila_model(args.vila_model_path)
    elif args.metric == "dino":
        dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        dino_model.eval()
        dino_model.to(device)
        dino_transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        

    image_folder=os.path.join(outpath,'samples')
    start_folder=os.path.join(outpath,'reference')
    target_folder=os.path.join(outpath,'target')
    
    file_names = os.listdir(image_folder)
    file_names.sort(key=lambda x: int(x.split("_")[-1].split('.')[0]))  # sort
    counter_ids = [int(x.split("_")[-1].split('.')[0]) for x in file_names]

    cnt = 0
    total = []

    for file_name, i in zip(file_names, counter_ids):
        image_path = os.path.join(image_folder,file_name)
        target_image_path = os.path.join(target_folder,file_name)
        start_image_path = os.path.join(start_folder,file_name)
        # image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        if args.prompt_in_filename:
            prompt = file_name.split("_")[0]
        else:
            prompt_path = os.path.join(outpath, "prompts.json")
            with open(prompt_path, 'r') as f:
                prompt_list = json.load(f)
            prompt = prompt_list[i]
            
        with torch.no_grad():
            if args.metric == "imagereward":
                reward = model.score(prompt, [image_path])
            elif args.metric in ["picscore", "clip", "aesthetic"]:
                reward = model.score(Image.open(image_path), prompt)[0]
            elif  args.metric == "hpsv2":
                reward = model.score([image_path], prompt)[0].item()
            elif args.metric in ["topiq","musiq","liqe","psnr","lpips","ssim"]:
                reward =  model.score([target_image_path],[image_path])[0]
            elif args.metric == "vila":
                image = preprocess_image(image_path, _PRE_CROP_SIZE, _IMAGE_SIZE)
                input_batch = NestedMap(
                    image=image,
                    ids=jnp.zeros((1, 1, _MAX_TEXT_LEN), dtype=jnp.int32),
                    paddings=jnp.zeros((1, 1, _MAX_TEXT_LEN), dtype=jnp.int32),
                )

                context_p = base_layer.JaxContext.HParams(do_eval=True)
                with base_layer.JaxContext(context_p):
                    predictions = model.apply(
                        {'params': model_states.mdl_vars['params']},
                        input_batch,
                        method=model.compute_predictions,
                    )
                    reward = predictions['quality_scores'][0][0].item()
            elif args.metric == "clipi":
                image_pairs=[[Image.open(target_image_path),  # gt
                      Image.open(image_path).convert('RGB'),  # output
                     None]]
                reward = eval_clip_i(image_pairs=image_pairs, model=clip_model, transform=transform, url_flag=False)
                
            elif args.metric == "dino":
                reward = eval_clip_i(image_pairs=image_pairs, model=dino_model, transform=dino_transform,
                                         url_flag=False, metric='dino')


        total.append(reward)
        cnt += 1
        if (cnt % 100 == 0):
            print(f"{args.metric}:{cnt} prompt(s) have been processed!")


    sim_dict=[]
    for i in range(len(total)):
        tmp={}
        tmp['question_id']=i
        tmp["answer"] = total[i]
        sim_dict.append(tmp)
    
    json_file = json.dumps(sim_dict)
    savepath = os.path.join(outpath,f"annotation_{args.metric}")
    os.makedirs(savepath, exist_ok=True)
    with open(f'{savepath}/vqa_result.json', 'w') as f:
        f.write(json_file)
    print(f"save to {savepath}")

    # score avg
    score=0
    for i in range(len(sim_dict)):
        score+=float(sim_dict[i]['answer'])
    with open(f'{savepath}/score_avg.txt', 'w') as f:
        f.write('score avg:'+str(score/len(sim_dict)))
    print("score avg:", score/len(sim_dict))



if __name__ == "__main__":
    args = parse_args()
    main(args)


