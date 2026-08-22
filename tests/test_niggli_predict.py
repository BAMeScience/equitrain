from equitrain import (
    get_args_parser_evaluate,
    get_args_parser_predict,
    get_args_parser_preprocess,
    get_args_parser_train,
)


def _parser_has_dest(parser, dest):
    return any(action.dest == dest for action in parser._actions)


def test_niggli_reduce_is_preprocess_only():
    assert _parser_has_dest(get_args_parser_preprocess(), 'niggli_reduce')
    assert not _parser_has_dest(get_args_parser_train(), 'niggli_reduce')
    assert not _parser_has_dest(get_args_parser_evaluate(), 'niggli_reduce')
    assert not _parser_has_dest(get_args_parser_predict(), 'niggli_reduce')
