from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np

from color_correction import wb_autocorrect, wb_from_ref


def parse_args():
    parser = ArgumentParser("White-balance app")

    parser.add_argument("src", type=Path, help="Source image")
    parser.add_argument("--ref", "-r", type=Path, help="Reference image")

    return parser.parse_args()


def main():
    args = parse_args()
    print(args)

    assert args.src.exists(), "Source image does not exist"
    assert args.ref is None or args.ref.exists(), "Referece image does not exist"
    
    source = cv2.imread(str(args.src), cv2.IMREAD_UNCHANGED)
    if args.ref is not None:
        reference = cv2.imread(str(args.ref), cv2.IMREAD_UNCHANGED)
        output = wb_from_ref(source, reference)
        out_path = args.src.parent / f"MATCHED_{args.src.stem}_to_{args.ref.stem}.png"
        print("Saving image to", out_path)
        cv2.imwrite(str(out_path), output)
    else:
        output = wb_autocorrect(source)
        out_path = args.src.parent / f"AWB_{args.src.stem}.png"
        print("Saving image to", out_path)
        cv2.imwrite(str(out_path), output)


if __name__ == '__main__':
    main()