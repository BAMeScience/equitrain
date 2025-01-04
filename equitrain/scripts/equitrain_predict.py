#! /usr/bin/env python

import sys

from equitrain import get_args_parser_predict
from equitrain import predict

# %%

def main():

    parser = get_args_parser_predict()

    try:
        predictions = predict(parser.parse_args())

        # TODO: Do something more useful with the result
        print(predictions)

    except ValueError as v:
        print(v, file=sys.stderr)
        exit(1)

# %%
if __name__ == "__main__":
    main()
