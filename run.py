import argparse

from run_map import main as run_map

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default=".", help="Path to gt and prediction data to test.")
    parser.add_argument('--rotated', type=str, default=True, help="If bounding boxes are rotated.")
    parser.add_argument('--verbose', type=bool, default=False, help="Print the associated gt with detection.")
    parser.add_argument('--overlap', default=0.5, type=float, help="Print the associated gt with detection.")
    parser.add_argument('--epochs', default=1, type=int, help="Number of epochs in training experiment.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    best_map = 0.0
    best_map_epoch = 0
    for epoch in range(int(args.epochs)):
        input_path = args.input_path + str(epoch)
        map = run_map(input_path, args.overlap)
        if map > best_map:
            best_map = map
            best_map_epoch = epoch
        print(f'EPOCH {epoch} mAP = {map}')

    print(f"\n Best mAP: {best_map} on epoch {best_map_epoch}")