import os, argparse
import datetime

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import logging
from accelerate.utils import DistributedDataParallelKwargs

# from trainer import Trainer

def make_args_parser():
    parser = argparse.ArgumentParser("MeshAnything", add_help=False)

    ##### Setting #####
    parser.add_argument("--input_pc_num", default=8192, type=int)
    parser.add_argument("--max_vertices", default=4000, type=int)

    parser.add_argument("--warm_lr_epochs", default=1, type=int)
    parser.add_argument("--num_beams", default=5, type=int)
    parser.add_argument("--top_k", default=30, type=int)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--temp", default=0.3, type=float)
    parser.add_argument("--face_threshold", default=0.65, type=float)

    ##### Model Setups #####
    parser.add_argument(
        '--pretrained_tokenizer_weight',
        default=None,
        type=str,
        help="The weight for pre-trained vqvae"
    )

    parser.add_argument('--llm', default="facebook/opt-350m", type=str, help="The LLM backend")
    parser.add_argument("--gen_n_max_triangles", default=4000, type=int, help="max number of triangles")

    ##### Training #####
    parser.add_argument("--eval_every_iteration", default=2000, type=int)
    parser.add_argument("--save_every", default=1, type=int)
    parser.add_argument("--generate_every_data", default=100, type=int)

    parser.add_argument("--clip_gradient", default=1., type=float, help="Max L2 norm of the gradient")
    parser.add_argument("--n_discrete_size", default=128, type=int, help="discretized 3D space")
    parser.add_argument('--model_name', default="MeshGen", type=str)
    parser.add_argument("--layer_num", default=24, type=int)

    parser.add_argument("--base_lr", default=1e-4, type=float)
    parser.add_argument("--final_lr", default=1e-6, type=float)
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--optimizer", default="AdamW", type=str)
    parser.add_argument("--warm_lr", default=1e-6, type=float)

    parser.add_argument("--checkpoint_dir", default="Training", type=str)
    parser.add_argument("--log_every", default=10, type=int)
    parser.add_argument("--test_only", default=False, action="store_true")
    parser.add_argument("--use_trainset", default=False, action="store_true")
    parser.add_argument("--train_plus", default=False, action="store_true")
    parser.add_argument("--generate_every_iteration", default=18000, type=int) #18000

    parser.add_argument("--max_epoch", default=100, type=int)
    parser.add_argument("--start_eval_after", default=-1, type=int)
    parser.add_argument("--precision", default="fp16", type=str)
    
    parser.add_argument("--num_workers", default=8, type=int)

    parser.add_argument('--pretrained_weight', default=None, type=str)
    
    #Defualt Setting 
    parser.add_argument("--shift_scale", default=0.1, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)

    ##### Demo #####
    parser.add_argument("--mesh_path", default="assets",  type=str)
    parser.add_argument("--render_process", default=False, action="store_true")
    parser.add_argument("--vert_model", default='VertGen', type=str)
    parser.add_argument("--face_model", default='FaceGen', type=str)
    parser.add_argument("--vertgen_ckpt", default=None, type=str)
    parser.add_argument("--facegen_ckpt", default=None, type=str)
    parser.add_argument("--use_refine", default=False, action="store_true")
    parser.add_argument("--variant", default='V1K', choices=['V1K', 'V4K']) 
    parser.add_argument("--outputs", default="results", type=str)
    parser.add_argument("--batch_size", default=2, type=int)

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = get_logger(__file__)

    args = make_args_parser()

    cur_time = datetime.datetime.now().strftime("%d_%H-%M-%S")
    wandb_name = args.checkpoint_dir + "_" +cur_time
    args.checkpoint_dir = os.path.join("gpt_out", wandb_name)
    print("checkpoint_dir:", args.checkpoint_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.precision,
        log_with="wandb",
        project_dir=args.checkpoint_dir,
        kwargs_handlers=[kwargs]
    )
    if "default" not in args.checkpoint_dir:
        accelerator.init_trackers(
            project_name=args.project_name,
            config=vars(args),
            init_kwargs={"wandb": {"name": wandb_name}}
        )

    set_seed(args.seed, device_specific=True)
    
    # trainer = Trainer(
    #     args,
    #     logger,
    #     accelerator,
    #     is_train = not args.test_only and not args.demo
    # )

    # trainer.train()

    
