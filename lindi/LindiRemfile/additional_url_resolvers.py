from typing import Callable, List


_global = {
    'additional_url_resolvers': []
}


def get_additional_url_resolvers() -> List[Callable[[str], str]]:
    return _global['additional_url_resolvers']


def add_additional_url_resolver(resolver: Callable[[str], str]):
    _global['additional_url_resolvers'].append(resolver)
