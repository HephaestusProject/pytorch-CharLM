"""CharLM: Character-Aware Neural Language Models

Usage:
    main.py <command> [<args>...]
    main.py (-h | --help)

Available commands:
    build-vocabulary
    train
    predict
    test

Options:
    -h --help     Show this.

See 'python main.py <command> --help' for more information on a specific command.
"""
from type_docopt import docopt
from pathlib import Path

from utils import parse

if __name__ == "__main__":
    args = docopt(__doc__, options_first=True)
    argv = [args["<command>"]] + args["<args>"]

    if args["<command>"] == "build-vocabulary":
        from build_vocabulary import __doc__, build_vocabulary

        build_vocabulary(docopt(__doc__, argv=argv, types={"path": Path}))
    elif args["<command>"] == "train":
        from train import __doc__, train

        train(docopt(__doc__, argv=argv, types={"path": Path}))
    elif args["<command>"] == "predict":
        from predict import __doc__, predict

        predict(docopt(__doc__, argv=argv, types={"path": Path}))
    elif args["<command>"] == "test":
        from test import __doc__, test

        test(docopt(__doc__, argv=argv, types={"path": Path}))
    else:
        raise NotImplementedError(f"Command does not exist: {args['<command>']}")
