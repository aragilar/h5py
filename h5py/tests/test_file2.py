# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Tests the h5py.File object.
"""

from h5py import h5f
from h5py import File, register_driver, registered_drivers, unregister_driver
from h5py._hl.files import _drivers

from .common import TestCase

import io
from tempfile import NamedTemporaryFile
import os


def nfiles():
    return h5f.get_obj_count(h5f.OBJ_ALL, h5f.OBJ_FILE)

def ngroups():
    return h5f.get_obj_count(h5f.OBJ_ALL, h5f.OBJ_GROUP)


class TestDealloc(TestCase):

    """
        Behavior on object deallocation.  Note most of this behavior is
        delegated to FileID.
    """

    def test_autoclose(self):
        """ File objects close automatically when out of scope, but
        other objects remain open. """

        start_nfiles = nfiles()
        start_ngroups = ngroups()

        fname = self.mktemp()
        f = File(fname, 'w')
        g = f['/']

        self.assertEqual(nfiles(), start_nfiles+1)
        self.assertEqual(ngroups(), start_ngroups+1)

        del f

        self.assertTrue(g)
        self.assertEqual(nfiles(), start_nfiles)
        self.assertEqual(ngroups(), start_ngroups+1)

        f = g.file

        self.assertTrue(f)
        self.assertEqual(nfiles(), start_nfiles+1)
        self.assertEqual(ngroups(), start_ngroups+1)

        del g

        self.assertEqual(nfiles(), start_nfiles+1)
        self.assertEqual(ngroups(), start_ngroups)

        del f

        self.assertEqual(nfiles(), start_nfiles)
        self.assertEqual(ngroups(), start_ngroups)


class TestDriverRegistration(TestCase):
    def test_register_driver(self):
        called_with = [None]

        def set_fapl(plist, *args, **kwargs):
            called_with[0] = args, kwargs
            return _drivers['sec2'](plist)

        register_driver('new-driver', set_fapl)
        self.assertIn('new-driver', registered_drivers())

        fname = self.mktemp()
        File(fname, driver='new-driver', driver_arg_0=0, driver_arg_1=1,
                  mode='w')

        self.assertEqual(
            called_with,
            [((), {'driver_arg_0': 0, 'driver_arg_1': 1})],
        )

    def test_unregister_driver(self):
        register_driver('new-driver', lambda plist: None)
        self.assertIn('new-driver', registered_drivers())

        unregister_driver('new-driver')
        self.assertNotIn('new-driver', registered_drivers())

        with self.assertRaises(ValueError) as e:
            fname = self.mktemp()
            File(fname, driver='new-driver', mode='w')

        self.assertEqual(str(e.exception), 'Unknown driver type "new-driver"')


class TestCache(TestCase):
    def test_defaults(self):
        self.assertEqual(list(self.f.id.get_access_plist().get_cache()),
                         [0, 521, 1048576, 0.75])

    def test_nbytes(self):
        fname = self.mktemp()
        f = File(fname, 'w', rdcc_nbytes=1024)
        self.assertEqual(list(f.id.get_access_plist().get_cache()),
                         [0, 521, 1024, 0.75])

    def test_nslots(self):
        fname = self.mktemp()
        f = File(fname, 'w', rdcc_nslots=125)
        self.assertEqual(list(f.id.get_access_plist().get_cache()),
                         [0, 125, 1048576, 0.75])

    def test_w0(self):
        fname = self.mktemp()
        f = File(fname, 'w', rdcc_w0=0.25)
        self.assertEqual(list(f.id.get_access_plist().get_cache()),
                         [0, 521, 1048576, 0.25])


class TestFileObj(TestCase):

    def check_write(self, fileobj):
        f = File(fileobj, 'w')
        self.assertEqual(f.driver, 'fileobj')
        self.assertEqual(f.filename, repr(fileobj))
        f.create_dataset('test', data=list(range(12)))
        self.assertEqual(list(f), ['test'])
        self.assertEqual(list(f['test'][:]), list(range(12)))
        f.close()

    def check_read(self, fileobj):
        f = File(fileobj, 'r')
        self.assertEqual(list(f), ['test'])
        self.assertEqual(list(f['test'][:]), list(range(12)))
        self.assertRaises(Exception, f.create_dataset, 'another.test', data=list(range(3)))
        f.close()

    def test_BytesIO(self):
        with io.BytesIO() as fileobj:
            self.assertEqual(len(fileobj.getvalue()), 0)
            self.check_write(fileobj)
            self.assertGreater(len(fileobj.getvalue()), 0)
            self.check_read(fileobj)

    def test_file(self):
        fname = self.mktemp()
        try:
            with open(fname, 'wb+') as fileobj:
                self.assertEqual(os.path.getsize(fname), 0)
                self.check_write(fileobj)
                self.assertGreater(os.path.getsize(fname), 0)
                self.check_read(fileobj)
            with open(fname, 'rb') as fileobj:
                self.check_read(fileobj)
        finally:
            os.remove(fname)

    def test_TemporaryFile(self):
        # in this test, we check explicitly that temp file gets
        # automatically deleted upon h5py.File.close()...
        fileobj = NamedTemporaryFile()
        fname = fileobj.name
        f = File(fileobj, 'w')
        del fileobj
        # ... but in your code feel free to simply
        # f = h5py.File(TemporaryFile())

        f.create_dataset('test', data=list(range(12)))
        self.assertEqual(list(f), ['test'])
        self.assertEqual(list(f['test'][:]), list(range(12)))
        self.assertTrue(os.path.isfile(fname))
        f.close()
        self.assertFalse(os.path.isfile(fname))

    def test_exception_open(self):
        self.assertRaises(Exception, File, None,
                          driver='fileobj', mode='x')
        self.assertRaises(Exception, File, 'rogue',
                          driver='fileobj', mode='x')
        self.assertRaises(Exception, File, self,
                          driver='fileobj', mode='x')

    def test_exception_read(self):

        class BrokenBytesIO(io.BytesIO):
            def readinto(self, b):
                raise Exception('I am broken')

        f = File(BrokenBytesIO(), 'w')
        f.create_dataset('test', data=list(range(12)))
        self.assertRaises(Exception, list, f['test'])

    def test_exception_write(self):

        class BrokenBytesIO(io.BytesIO):
            def write(self, b):
                raise Exception('I am broken')

        f = File(BrokenBytesIO(), 'w')
        self.assertRaises(Exception, f.create_dataset, 'test',
                          data=list(range(12)))
        self.assertRaises(Exception, f.close)

    def test_exception_close(self):
        fileobj = io.BytesIO()
        f = File(fileobj, 'w')
        fileobj.close()
        self.assertRaises(Exception, f.close)

    def test_method_vanish(self):
        fileobj = io.BytesIO()
        f = File(fileobj, 'w')
        f.create_dataset('test', data=list(range(12)))
        self.assertEqual(list(f['test'][:]), list(range(12)))
        fileobj.readinto = None
        self.assertRaises(Exception, list, f['test'])


class TestTrackOrder(TestCase):
    def populate(self, f):
        for i in range(100):
            # Mix group and dataset creation.
            if i % 10 == 0:
                f.create_group(str(i))
            else:
                f[str(i)] = [i]

    def test_track_order(self):
        fname = self.mktemp()
        f = File(fname, 'w', track_order=True)  # creation order
        self.populate(f)
        self.assertEqual(list(f),
                         [str(i) for i in range(100)])

    def test_no_track_order(self):
        fname = self.mktemp()
        f = File(fname, 'w', track_order=False)  # name alphanumeric
        self.populate(f)
        self.assertEqual(list(f),
                         sorted([str(i) for i in range(100)]))
