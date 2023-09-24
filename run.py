import sys
import argparse

from forget_me_not.run_exps import text_exps
from forget_me_not.run_exps import img_exps

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--report_root_dir", type=str, required=True)
    parser.add_argument("--max_epochs", type=int, required=True)
    parser.add_argument("--accelerator", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=False, default=None)
    parser.add_argument("--only_eval", type=bool, required=False, default=False)
    parser.add_argument("--seed", type=int, required=False, default=0)

    
    args = parser.parse_args(sys.argv[1:])
    if args.exp == 'text':
        text_exps.main(args)
    elif args.exp == 'img':
        img_exps.main(args)
    else:
        raise ValueError(f"Unknown experiment: {args.exp}")