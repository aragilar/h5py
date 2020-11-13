import pytest

from .common import TestCase


class TestException(Exception):
    pass

def throwing(name, obj):
    print(name, obj)
    raise TestException("throwing exception")

class TestVisit(TestCase):
    def test_visit(self):
        fname = self.mktemp()
        self.f.create_dataset('foo', (100,), dtype='uint8')
        with pytest.raises(TestException, match='throwing exception'):
            self.f.visititems(throwing)
