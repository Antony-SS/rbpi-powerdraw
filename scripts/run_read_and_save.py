from experiments.impl.read_and_save import ReadAndSave
import picamera2
from argparse import ArgumentParser
import importlib

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--size", type=str, required=True)
    parser.add_argument("--experiment_duration", type=int, required=False, default=60)
    parser.add_argument("--output_dir", type=str, required=False)
    return parser.parse_args()

def parse_size(size_str: str):
    return tuple(int(x) for x in size_str.split("x"))

def main():

    args = parse_args()
    size = parse_size(args.size)
    output_dir = args.output_dir if args.output_dir is not None else f"output/{ReadAndSave.NAME}_fps{args.fps}_size{size[0]}x{size[1]}"
    read_and_save = ReadAndSave(picamera2.Picamera2(), args.fps, size, output_dir, experiment_duration=args.experiment_duration)
    read_and_save.run_experiment()

if __name__ == "__main__":
    main()