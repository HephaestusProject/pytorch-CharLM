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
from pathlib import Path

from type_docopt import docopt


class IntList(list):
    def __init__(self, arg):
        int_list = [int(value) for value in arg.split(",")]
        super(IntList, self).__init__(int_list)


if __name__ == "__main__":
    args = docopt(__doc__, options_first=True)
    argv = [args["<command>"]] + args["<args>"]

    if args["<command>"] == "build-vocabulary":
        from build_vocabulary import __doc__, build_vocabulary

        build_vocabulary(docopt(__doc__, argv=argv, types={"path": Path}))

    elif args["<command>"] == "train":
        from train import __doc__, train

        train(docopt(__doc__, argv=argv, types={"path": Path, "IntList": IntList}))


    else:
        raise NotImplementedError(f"Command does not exist: {args['<command>']}")
