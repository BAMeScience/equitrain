import sys

from .. import evaluate, get_args_parser_evaluate


# %%
def main():
    parser = get_args_parser_evaluate()

    try:
        evaluate(parser.parse_args())

    except ValueError as v:
        print(v, file=sys.stderr)
        sys.exit(1)


# %%
if __name__ == '__main__':
    main()
