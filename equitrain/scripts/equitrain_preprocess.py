import sys

from .. import get_args_parser_preprocess, preprocess


# %%
def main():
    parser = get_args_parser_preprocess()

    try:
        preprocess(parser.parse_args())
    except ValueError as v:
        print(v, file=sys.stderr)
        sys.exit(1)


# %%
if __name__ == '__main__':
    main()
