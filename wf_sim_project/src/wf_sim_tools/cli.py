import argparse
import sys
from .api import run_sims

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Required
    parser.add_argument("--n-sims", type=int, required=True)
    parser.add_argument("--pop-size", type=int, required=True)
    
    # Optional
    parser.add_argument("--theta", type=float, default=50.0)
    parser.add_argument("--rho", type=float, default=50.0)
    parser.add_argument("--run-mult", type=int, default=2)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.0)
    parser.add_argument("--symmetry", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--polygenic", action="store_true")
    parser.add_argument("--demography-file", type=str, default=None)
    parser.add_argument("-o", "--output", type=str, help="Save to file instead of printing")

    args = parser.parse_args()

    try:
        # Call the Python API
        df = run_sims(
            n_sims=args.n_sims, pop_size=args.pop_size, theta=args.theta,
            rho=args.rho, run_mult=args.run_mult, beta1=args.beta1,
            beta2=args.beta2, symmetry=args.symmetry, seed=args.seed,
            polygenic=args.polygenic, demography_file=args.demography_file
        )

        if args.output:
            df.to_csv(args.output, sep="\t", index=False)
            print(f"Saved to {args.output}", file=sys.stderr)
        else:
            # Print to stdout in tab-separated format
            df.to_csv(sys.stdout, sep="\t", index=False)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()