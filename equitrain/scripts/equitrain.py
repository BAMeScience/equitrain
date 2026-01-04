import sys

from .. import ArgumentError, get_args_parser_train, train


# %%
def main():
    parser = get_args_parser_train()

    try:
        train(parser.parse_args())
    except ArgumentError as v:
        print(v, file=sys.stderr)
        sys.exit(1)


# %%
if __name__ == '__main__':
    main()
