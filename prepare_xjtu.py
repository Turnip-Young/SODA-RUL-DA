import argparse
import time

from dataset_xjtu import BEARING_META, load_or_build_bearing


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../XJTU-SY_Bearing_Datasets")
    parser.add_argument("--cache_root", type=str, default="cache/xjtu_features")
    parser.add_argument("--bearings", type=str, default="all", help="comma-separated bearing names or all")
    parser.add_argument("--rebuild", type=int, default=0, choices=[0, 1])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    t0 = time.perf_counter()
    if args.bearings == "all":
        bearings = sorted(BEARING_META)
    else:
        bearings = [i.strip() for i in args.bearings.split(",") if i.strip()]
    for bearing in bearings:
        x, y = load_or_build_bearing(args.data_root, args.cache_root, bearing, bool(args.rebuild))
        print("{}: x={}, y=[{:.3f}->{:.3f}]".format(bearing, x.shape, float(y[0]), float(y[-1])))
    t1 = time.perf_counter()
    print("done in {:.2f}s".format(t1 - t0))
