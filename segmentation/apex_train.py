from simplecv import apex_ddp_train as train
from data import thyroid
from module import foroptFPN
import torch

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    args = train.parser.parse_args()
    print("Following are args.=================================")    
    print(args)
    print(args.local_rank)
    print(args.config_path)
    print("args end.===========================================")
    train.run(local_rank=args.local_rank,
              config_path=args.config_path,
              model_dir=args.model_dir,
              opt_level=args.opt_level,
              cpu_mode=args.cpu,
              after_construct_launcher_callbacks=[],
              opts=args.opts)
