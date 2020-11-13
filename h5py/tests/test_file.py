# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    File object test module.

    Tests all aspects of File objects, including their creation.
"""

import io
import os
import pathlib
import pickle
import stat
import tempfile

import pytest

import h5py
from h5py import File
from h5py._hl.files import _drivers
from .common import ut, TestCase, UNICODE_FILENAMES, closed_tempfile


def nfiles():
    return h5py.h5f.get_obj_count(h5py.h5f.OBJ_ALL, h5py.h5f.OBJ_FILE)


def ngroups():
    return h5py.h5f.get_obj_count(h5py.h5f.OBJ_ALL, h5py.h5f.OBJ_GROUP)


class TestFileOpen(TestCase):

    """
        Feature: Opening files with Python-style modes.
    """

    def test_default(self):
        """ Default semantics in the presence or absence of a file """
        fname = self.mktemp()

        # No existing file; error
        with pytest.raises(OSError):
            with File(fname):
                pass


        # Existing readonly file; open read-only
        with File(fname, 'w'):
            pass
        os.chmod(fname, stat.S_IREAD)
        try:
            with File(fname) as f:
                self.assertTrue(f)
                self.assertEqual(f.mode, 'r')
        finally:
            os.chmod(fname, stat.S_IWRITE)

        # File exists but is not HDF5; raise IOError
        with open(fname, 'wb') as f:
            f.write(b'\x00')
        with self.assertRaises(IOError):
            File(fname)

    def test_create(self):
        """ Mode 'w' opens file in overwrite mode """
        fname = self.mktemp()
        fid = File(fname, 'w')
        self.assertTrue(fid)
        fid.create_group('foo')
        fid.close()
        fid = File(fname, 'w')
        self.assertNotIn('foo', fid)
        fid.close()

    def test_create_exclusive(self):
        """ Mode 'w-' opens file in exclusive mode """
        fname = self.mktemp()
        fid = File(fname, 'w-')
        self.assertTrue(fid)
        fid.close()
        with self.assertRaises(IOError):
            File(fname, 'w-')

    def test_append(self):
        """ Mode 'a' opens file in append/readwrite mode, creating if necessary """
        fname = self.mktemp()
        fid = File(fname, 'a')
        try:
            self.assertTrue(fid)
            fid.create_group('foo')
            assert 'foo' in fid
        finally:
            fid.close()
        fid = File(fname, 'a')
        try:
            assert 'foo' in fid
            fid.create_group('bar')
            assert 'bar' in fid
        finally:
            fid.close()

    def test_readonly(self):
        """ Mode 'r' opens file in readonly mode """
        fname = self.mktemp()
        fid = File(fname, 'w')
        fid.close()
        self.assertFalse(fid)
        fid = File(fname, 'r')
        self.assertTrue(fid)
        with self.assertRaises(ValueError):
            fid.create_group('foo')
        fid.close()

    def test_readwrite(self):
        """ Mode 'r+' opens existing file in readwrite mode """
        fname = self.mktemp()
        fid = File(fname, 'w')
        fid.create_group('foo')
        fid.close()
        fid = File(fname, 'r+')
        assert 'foo' in fid
        fid.create_group('bar')
        assert 'bar' in fid
        fid.close()

    def test_nonexistent_file(self):
        """ Modes 'r' and 'r+' do not create files """
        fname = self.mktemp()
        with self.assertRaises(IOError):
            File(fname, 'r')
        with self.assertRaises(IOError):
            File(fname, 'r+')

    def test_invalid_mode(self):
        """ Invalid modes raise ValueError """
        with self.assertRaises(ValueError):
            File(self.mktemp(), 'mongoose')

@ut.skipIf(h5py.version.hdf5_version_tuple < (1, 10, 1),
               'Requires HDF5 1.10.1 or later')
class TestSpaceStrategy(TestCase):

    """
        Feature: Create file with specified file space strategy
    """

    def test_create_with_space_strategy(self):
        """ Create file with file space strategy """
        fname = self.mktemp()
        fid = File(fname, 'w', fs_strategy="page",
                   fs_persist=True, fs_threshold=100)
        self.assertTrue(fid)
        # Unable to set file space strategy of an existing file
        with self.assertRaises(ValueError):
            File(fname, 'a', fs_strategy="page")
        # Invalid file space strategy type
        with self.assertRaises(ValueError):
            File(self.mktemp(), 'w', fs_strategy="invalid")

        dset = fid.create_dataset('foo', (100,), dtype='uint8')
        dset[...] = 1
        dset = fid.create_dataset('bar', (100,), dtype='uint8')
        dset[...] = 1
        del fid['foo']
        fid.close()

        fid = File(fname, 'a')
        plist = fid.id.get_create_plist()
        fs_strat = plist.get_file_space_strategy()
        assert(fs_strat[0] == 1)
        assert(fs_strat[1] == True)
        assert(fs_strat[2] == 100)

        dset = fid.create_dataset('foo2', (100,), dtype='uint8')
        dset[...] = 1
        fid.close()

class TestModes(TestCase):

    """
        Feature: File mode can be retrieved via file.mode
    """

    def test_mode_attr(self):
        """ Mode equivalent can be retrieved via property """
        fname = self.mktemp()
        with File(fname, 'w') as f:
            self.assertEqual(f.mode, 'r+')
        with File(fname, 'r') as f:
            self.assertEqual(f.mode, 'r')

    def test_mode_external(self):
        """ Mode property works for files opened via external links

        Issue 190.
        """
        fname1 = self.mktemp()
        fname2 = self.mktemp()

        f1 = File(fname1, 'w')
        f1.close()

        f2 = File(fname2, 'w')
        try:
            f2['External'] = h5py.ExternalLink(fname1, '/')
            f3 = f2['External'].file
            self.assertEqual(f3.mode, 'r+')
        finally:
            f2.close()
            f3.close()

        f2 = File(fname2, 'r')
        try:
            f3 = f2['External'].file
            self.assertEqual(f3.mode, 'r')
        finally:
            f2.close()
            f3.close()


class TestDrivers(TestCase):

    """
        Feature: Files can be opened with low-level HDF5 drivers. Does not
        include MPI drivers (see bottom).
    """

    @ut.skipUnless(os.name == 'posix', "Stdio driver is supported on posix")
    def test_stdio(self):
        """ Stdio driver is supported on posix """
        fid = File(self.mktemp(), 'w', driver='stdio')
        self.assertTrue(fid)
        self.assertEqual(fid.driver, 'stdio')
        fid.close()

    @ut.skipUnless(os.name == 'posix', "Sec2 driver is supported on posix")
    def test_sec2(self):
        """ Sec2 driver is supported on posix """
        fid = File(self.mktemp(), 'w', driver='sec2')
        self.assertTrue(fid)
        self.assertEqual(fid.driver, 'sec2')
        fid.close()

    def test_core(self):
        """ Core driver is supported (no backing store) """
        fname = self.mktemp()
        fid = File(fname, 'w', driver='core', backing_store=False)
        self.assertTrue(fid)
        self.assertEqual(fid.driver, 'core')
        fid.close()
        self.assertFalse(os.path.exists(fname))

    def test_backing(self):
        """ Core driver saves to file when backing store used """
        fname = self.mktemp()
        fid = File(fname, 'w', driver='core', backing_store=True)
        fid.create_group('foo')
        fid.close()
        fid = File(fname, 'r')
        assert 'foo' in fid
        fid.close()
        # keywords for other drivers are invalid when using the default driver
        with self.assertRaises(TypeError):
            File(fname, 'w', backing_store=True)

    def test_readonly(self):
        """ Core driver can be used to open existing files """
        fname = self.mktemp()
        fid = File(fname, 'w')
        fid.create_group('foo')
        fid.close()
        fid = File(fname, 'r', driver='core')
        self.assertTrue(fid)
        assert 'foo' in fid
        with self.assertRaises(ValueError):
            fid.create_group('bar')
        fid.close()

    def test_blocksize(self):
        """ Core driver supports variable block size """
        fname = self.mktemp()
        fid = File(fname, 'w', driver='core', block_size=1024,
                   backing_store=False)
        self.assertTrue(fid)
        fid.close()

    def test_split(self):
        """ Split stores metadata in a separate file """
        fname = self.mktemp()
        fid = File(fname, 'w', driver='split')
        fid.close()
        self.assertTrue(os.path.exists(fname + '-m.h5'))
        fid = File(fname, 'r', driver='split')
        self.assertTrue(fid)
        fid.close()

    def test_fileobj(self):
        """ Python file object driver is supported """
        tf = tempfile.TemporaryFile()
        fid = File(tf, 'w', driver='fileobj')
        self.assertTrue(fid)
        self.assertEqual(fid.driver, 'fileobj')
        fid.close()
        # Driver must be 'fileobj' for file-like object if specified
        with self.assertRaises(ValueError):
            File(tf, 'w', driver='core')


    # TODO: family driver tests


@ut.skipUnless(h5py.version.hdf5_version_tuple < (1, 10, 2),
               'Requires HDF5 before 1.10.2')
class TestLibver(TestCase):

    """
        Feature: File format compatibility bounds can be specified when
        opening a file.
    """

    def test_default(self):
        """ Opening with no libver arg """
        f = File(self.mktemp(), 'w')
        self.assertEqual(f.libver, ('earliest', 'latest'))
        f.close()

    def test_single(self):
        """ Opening with single libver arg """
        f = File(self.mktemp(), 'w', libver='latest')
        self.assertEqual(f.libver, ('latest', 'latest'))
        f.close()

    def test_multiple(self):
        """ Opening with two libver args """
        f = File(self.mktemp(), 'w', libver=('earliest', 'latest'))
        self.assertEqual(f.libver, ('earliest', 'latest'))
        f.close()

    def test_none(self):
        """ Omitting libver arg results in maximum compatibility """
        f = File(self.mktemp(), 'w')
        self.assertEqual(f.libver, ('earliest', 'latest'))
        f.close()


@ut.skipIf(h5py.version.hdf5_version_tuple < (1, 10, 2),
           'Requires HDF5 1.10.2 or later')
class TestNewLibver(TestCase):

    """
        Feature: File format compatibility bounds can be specified when
        opening a file.

        Requirement: HDF5 1.10.2 or later
    """

    @classmethod
    def setUpClass(cls):
        super(TestNewLibver, cls).setUpClass()

        # Current latest library bound label
        if h5py.version.hdf5_version_tuple < (1, 11, 4):
            cls.latest = 'v110'
        else:
            cls.latest = 'v112'

    def test_default(self):
        """ Opening with no libver arg """
        f = File(self.mktemp(), 'w')
        self.assertEqual(f.libver, ('earliest', self.latest))
        f.close()

    def test_single(self):
        """ Opening with single libver arg """
        f = File(self.mktemp(), 'w', libver='latest')
        self.assertEqual(f.libver, (self.latest, self.latest))
        f.close()

    def test_single_v108(self):
        """ Opening with "v108" libver arg """
        f = File(self.mktemp(), 'w', libver='v108')
        self.assertEqual(f.libver, ('v108', self.latest))
        f.close()

    def test_single_v110(self):
        """ Opening with "v110" libver arg """
        f = File(self.mktemp(), 'w', libver='v110')
        self.assertEqual(f.libver, ('v110', self.latest))
        f.close()

    @ut.skipIf(h5py.version.hdf5_version_tuple < (1, 11, 4),
           'Requires HDF5 1.11.4 or later')
    def test_single_v112(self):
        """ Opening with "v112" libver arg """
        f = File(self.mktemp(), 'w', libver='v112')
        self.assertEqual(f.libver, ('v112', self.latest))
        f.close()

    def test_multiple(self):
        """ Opening with two libver args """
        f = File(self.mktemp(), 'w', libver=('earliest', 'v108'))
        self.assertEqual(f.libver, ('earliest', 'v108'))
        f.close()

    def test_none(self):
        """ Omitting libver arg results in maximum compatibility """
        f = File(self.mktemp(), 'w')
        self.assertEqual(f.libver, ('earliest', self.latest))
        f.close()


class TestUserblock(TestCase):

    """
        Feature: Files can be create with user blocks
    """

    def test_create_blocksize(self):
        """ User blocks created with w, w-, x and properties work correctly """
        f = File(self.mktemp(), 'w-', userblock_size=512)
        try:
            self.assertEqual(f.userblock_size, 512)
        finally:
            f.close()

        f = File(self.mktemp(), 'x', userblock_size=512)
        try:
            self.assertEqual(f.userblock_size, 512)
        finally:
            f.close()

        f = File(self.mktemp(), 'w', userblock_size=512)
        try:
            self.assertEqual(f.userblock_size, 512)
        finally:
            f.close()
        # User block size must be an integer
        with self.assertRaises(ValueError):
            File(self.mktemp(), 'w', userblock_size='non')


    def test_write_only(self):
        """ User block only allowed for write """
        name = self.mktemp()
        f = File(name, 'w')
        f.close()

        with self.assertRaises(ValueError):
            f = h5py.File(name, 'r', userblock_size=512)

        with self.assertRaises(ValueError):
            f = h5py.File(name, 'r+', userblock_size=512)

    def test_match_existing(self):
        """ User block size must match that of file when opening for append """
        name = self.mktemp()
        f = File(name, 'w', userblock_size=512)
        f.close()

        with self.assertRaises(ValueError):
            f = File(name, 'a', userblock_size=1024)

        f = File(name, 'a', userblock_size=512)
        try:
            self.assertEqual(f.userblock_size, 512)
        finally:
            f.close()

    def test_power_of_two(self):
        """ User block size must be a power of 2 and at least 512 """
        name = self.mktemp()

        with self.assertRaises(ValueError):
            f = File(name, 'w', userblock_size=128)

        with self.assertRaises(ValueError):
            f = File(name, 'w', userblock_size=513)

        with self.assertRaises(ValueError):
            f = File(name, 'w', userblock_size=1023)

    def test_write_block(self):
        """ Test that writing to a user block does not destroy the file """
        name = self.mktemp()

        f = File(name, 'w', userblock_size=512)
        f.create_group("Foobar")
        f.close()

        pyfile = open(name, 'r+b')
        try:
            pyfile.write(b'X' * 512)
        finally:
            pyfile.close()

        f = h5py.File(name, 'r')
        try:
            assert "Foobar" in f
        finally:
            f.close()

        pyfile = open(name, 'rb')
        try:
            self.assertEqual(pyfile.read(512), b'X' * 512)
        finally:
            pyfile.close()


class TestContextManager(TestCase):

    """
        Feature: File objects can be used as context managers
    """

    def test_context_manager(self):
        """ File objects can be used in with statements """
        with File(self.mktemp(), 'w') as fid:
            self.assertTrue(fid)
        self.assertTrue(not fid)


@ut.skipIf(not UNICODE_FILENAMES, "Filesystem unicode support required")
class TestUnicode(TestCase):

    """
        Feature: Unicode filenames are supported
    """

    def test_unicode(self):
        """ Unicode filenames can be used, and retrieved properly via .filename
        """
        fname = self.mktemp(prefix=chr(0x201a))
        fid = File(fname, 'w')
        try:
            self.assertEqual(fid.filename, fname)
            self.assertIsInstance(fid.filename, str)
        finally:
            fid.close()

    def test_unicode_hdf5_python_consistent(self):
        """ Unicode filenames can be used, and seen correctly from python
        """
        fname = self.mktemp(prefix=chr(0x201a))
        with File(fname, 'w') as f:
            self.assertTrue(os.path.exists(fname))

    def test_nonexistent_file_unicode(self):
        """
        Modes 'r' and 'r+' do not create files even when given unicode names
        """
        fname = self.mktemp(prefix=chr(0x201a))
        with self.assertRaises(IOError):
            File(fname, 'r')
        with self.assertRaises(IOError):
            File(fname, 'r+')


class TestFileProperty(TestCase):

    """
        Feature: A File object can be retrieved from any child object,
        via the .file property
    """

    def test_property(self):
        """ File object can be retrieved from subgroup """
        fname = self.mktemp()
        hfile = File(fname, 'w')
        try:
            hfile2 = hfile['/'].file
            self.assertEqual(hfile, hfile2)
        finally:
            hfile.close()

    def test_close(self):
        """ All retrieved File objects are closed at the same time """
        fname = self.mktemp()
        hfile = File(fname, 'w')
        grp = hfile.create_group('foo')
        hfile2 = grp.file
        hfile3 = hfile['/'].file
        hfile2.close()
        self.assertFalse(hfile)
        self.assertFalse(hfile2)
        self.assertFalse(hfile3)

    def test_mode(self):
        """ Retrieved File objects have a meaningful mode attribute """
        hfile = File(self.mktemp(), 'w')
        try:
            grp = hfile.create_group('foo')
            self.assertEqual(grp.file.mode, hfile.mode)
        finally:
            hfile.close()


class TestClose(TestCase):

    """
        Feature: Files can be closed
    """

    def test_close(self):
        """ Close file via .close method """
        fid = File(self.mktemp(), 'w')
        self.assertTrue(fid)
        fid.close()
        self.assertFalse(fid)

    def test_closed_file(self):
        """ Trying to modify closed file raises ValueError """
        fid = File(self.mktemp(), 'w')
        fid.close()
        with self.assertRaises(ValueError):
            fid.create_group('foo')

    def test_close_multiple_default_driver(self):
        fname = self.mktemp()
        f = h5py.File(fname, 'w')
        f.create_group("test")
        f.close()
        f.close()

class TestFlush(TestCase):

    """
        Feature: Files can be flushed
    """

    def test_flush(self):
        """ Flush via .flush method """
        fid = File(self.mktemp(), 'w')
        fid.flush()
        fid.close()


class TestRepr(TestCase):

    """
        Feature: File objects provide a helpful __repr__ string
    """

    def test_repr(self):
        """ __repr__ behaves itself when files are open and closed """
        fid = File(self.mktemp(), 'w')
        self.assertIsInstance(repr(fid), str)
        fid.close()
        self.assertIsInstance(repr(fid), str)


class TestFilename(TestCase):

    """
        Feature: The name of a File object can be retrieved via .filename
    """

    def test_filename(self):
        """ .filename behaves properly for string data """
        fname = self.mktemp()
        fid = File(fname, 'w')
        try:
            self.assertEqual(fid.filename, fname)
            self.assertIsInstance(fid.filename, str)
        finally:
            fid.close()


class TestCloseInvalidatesOpenObjectIDs(TestCase):

    """
        Ensure that closing a file invalidates object IDs, as appropriate
    """

    def test_close(self):
        """ Closing a file invalidates any of the file's open objects """
        with File(self.mktemp(), 'w') as f1:
            g1 = f1.create_group('foo')
            self.assertTrue(bool(f1.id))
            self.assertTrue(bool(g1.id))
            f1.close()
            self.assertFalse(bool(f1.id))
            self.assertFalse(bool(g1.id))
        with File(self.mktemp(), 'w') as f2:
            g2 = f2.create_group('foo')
            self.assertTrue(bool(f2.id))
            self.assertTrue(bool(g2.id))
            self.assertFalse(bool(f1.id))
            self.assertFalse(bool(g1.id))


class TestPathlibSupport(TestCase):

    """
        Check that h5py doesn't break on pathlib
    """
    def test_pathlib_accepted_file(self):
        """ Check that pathlib is accepted by h5py.File """
        with closed_tempfile() as f:
            path = pathlib.Path(f)
            with File(path, 'w') as f2:
                self.assertTrue(True)

    def test_pathlib_name_match(self):
        """ Check that using pathlib does not affect naming """
        with closed_tempfile() as f:
            path = pathlib.Path(f)
            with File(path, 'w') as h5f1:
                pathlib_name = h5f1.filename
            with File(f, 'w') as h5f2:
                normal_name = h5f2.filename
            self.assertEqual(pathlib_name, normal_name)


class TestPickle(TestCase):
    """Check that h5py.File can't be pickled"""
    def test_dump_error(self):
        with File(self.mktemp(), 'w') as f1:
            with self.assertRaises(TypeError):
                pickle.dumps(f1)


# unittest doesn't work with pytest fixtures (and possibly other features),
# hence no subclassing TestCase
@pytest.mark.mpi
class TestMPI:
    def test_mpio(self, mpi_file_name):
        """ MPIO driver and options """
        from mpi4py import MPI

        with File(mpi_file_name, 'w', driver='mpio', comm=MPI.COMM_WORLD) as f:
            assert f
            assert f.driver == 'mpio'

    @pytest.mark.skipif(h5py.version.hdf5_version_tuple < (1, 8, 9),
        reason="mpio atomic file operations were added in HDF5 1.8.9+")
    def test_mpi_atomic(self, mpi_file_name):
        """ Enable atomic mode for MPIO driver """
        from mpi4py import MPI

        with File(mpi_file_name, 'w', driver='mpio', comm=MPI.COMM_WORLD) as f:
            assert not f.atomic
            f.atomic = True
            assert f.atomic

    def test_close_multiple_mpio_driver(self, mpi_file_name):
        """ MPIO driver and options """
        from mpi4py import MPI

        f = File(mpi_file_name, 'w', driver='mpio', comm=MPI.COMM_WORLD)
        f.create_group("test")
        f.close()
        f.close()


@ut.skipIf(h5py.version.hdf5_version_tuple < (1, 10, 1),
               'Requires HDF5 1.10.1 or later')
class TestSWMRMode(TestCase):

    """
        Feature: Create file that switches on SWMR mode
    """

    def test_file_mode_generalizes(self):
        fname = self.mktemp()
        fid = File(fname, 'w', libver='latest')
        g = fid.create_group('foo')
        # fid and group member file attribute should have the same mode
        assert fid.mode == g.file.mode == 'r+'
        fid.swmr_mode = True
        # fid and group member file attribute should still be 'r+'
        # even though file intent has changed
        assert fid.mode == g.file.mode == 'r+'
        fid.close()

    def test_swmr_mode_consistency(self):
        fname = self.mktemp()
        fid = File(fname, 'w', libver='latest')
        g = fid.create_group('foo')
        assert fid.swmr_mode == g.file.swmr_mode == False
        fid.swmr_mode = True
        # This setter should affect both fid and group member file attribute
        assert fid.swmr_mode == g.file.swmr_mode == True
        fid.close()


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

        h5py.register_driver('new-driver', set_fapl)
        self.assertIn('new-driver', h5py.registered_drivers())

        fname = self.mktemp()
        File(fname, driver='new-driver', driver_arg_0=0, driver_arg_1=1,
                  mode='w')

        self.assertEqual(
            called_with,
            [((), {'driver_arg_0': 0, 'driver_arg_1': 1})],
        )

    def test_unregister_driver(self):
        h5py.register_driver('new-driver', lambda plist: None)
        self.assertIn('new-driver', h5py.registered_drivers())

        h5py.unregister_driver('new-driver')
        self.assertNotIn('new-driver', h5py.registered_drivers())

        with self.assertRaises(ValueError) as e:
            fname = self.mktemp()
            File(fname, driver='new-driver', mode='w')

        self.assertEqual(str(e.exception), 'Unknown driver type "new-driver"')


class TestCache(TestCase):
    def test_defaults(self):
        fname = self.mktemp()
        f = File(fname, 'w')
        self.assertEqual(list(f.id.get_access_plist().get_cache()),
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
        fileobj = tempfile.NamedTemporaryFile()
        fname = fileobj.name
        f = File(fileobj, 'w')
        del fileobj
        # ... but in your code feel free to simply
        # f = h5py.File(tempfile.TemporaryFile())

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
            allow_write = False
            def write(self, b):
                if self.allow_write:
                    return super().write(b)
                else:
                    raise Exception('I am broken')

        bio = BrokenBytesIO()
        f = File(bio, 'w')
        try:
            self.assertRaises(Exception, f.create_dataset, 'test',
                              data=list(range(12)))
        finally:
            # Un-break writing so we can close: errors while closing get messy.
            bio.allow_write = True
            f.close()

    @ut.skip("Incompletely closed files can cause segfaults")
    def test_exception_close(self):
        fileobj = io.BytesIO()
        f = File(fileobj, 'w')
        fileobj.close()
        self.assertRaises(Exception, f.close)

    def test_exception_writeonly(self):
        # HDF5 expects read & write access to a file it's writing;
        # check that we get the correct exception on a write-only file object.
        fileobj = open(os.path.join(self.tempdir, 'a.h5'), 'wb')
        with self.assertRaises(io.UnsupportedOperation):
            f = File(fileobj, 'w')
            group = f.create_group("group")
            group.create_dataset("data", data='foo', dtype=h5py.string_dtype())


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
