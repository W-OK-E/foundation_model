import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session_name', default='train_session')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--preprocessing', default='none')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--script', default='train.py')  # Name of training script

    args = parser.parse_args()

    train_cmd = (
        f"python {args.script} "
        f"--dataset {args.dataset} "
        f"--preprocessing {args.preprocessing} "
        f"--lr {args.lr} "
        f"--epochs {args.epochs} "
        f"--img_size {args.img_size}"
    )
    
    # Start a new tmux session and run the training command
    subprocess.run(f"tmux new-session -d -s {args.session_name} '{train_cmd}'", shell=True)
    print(f"Started tmux session '{args.session_name}' running command:\n{train_cmd}")

if __name__ == "__main__":
    main()
