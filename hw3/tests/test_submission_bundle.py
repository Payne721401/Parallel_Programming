import hashlib
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from submission_bundle import make_submission, submission_files


class SubmissionBundleTests(unittest.TestCase):
    def source(self, root, text='#include "ca.hpp"\n#include "outer.hpp"\n'):
        root.mkdir()
        (root / "hw3-1.cu").write_text(text, encoding="utf-8")
        (root / "outer.hpp").write_text('#include "nested/inner.cuh"\n#include <cuda_runtime.h>\n', encoding="utf-8")
        (root / "nested").mkdir()
        (root / "nested/inner.cuh").write_text('#include "leaf.h"\n', encoding="utf-8")
        (root / "nested/leaf.h").write_text('// a dependency\n', encoding="utf-8")
        for name in ("ca.hpp", "main.cu", "Makefile", "unused.hpp"):
            (root / name).write_text("must not be packaged", encoding="utf-8")
        (root / "tests").mkdir()
        (root / "tests/test.hpp").write_text("not submitted", encoding="utf-8")

    def test_nested_relative_includes_duplicates_cycles_and_comments(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "source"
            self.source(root, '# include "outer.hpp"\n#include "outer.hpp"\n'
                        '// #include "missing.hpp"\n/* #include "also-missing.hpp" */\n'
                        '#include "ca.hpp"\n#include <vector>\n')
            (root / "nested/leaf.h").write_text('#include "inner.cuh"\n', encoding="utf-8")
            self.assertEqual(submission_files(root), ("hw3-1.cu", "nested/inner.cuh", "nested/leaf.h", "outer.hpp"))

    def test_archive_has_student_root_only_complete_headers_and_matching_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root, archive_path = Path(temp) / "source", Path(temp) / "submission.tar"
            self.source(root)
            manifest = make_submission(root, archive_path, "r13229005")
            self.assertEqual(set(manifest), set(submission_files(root)))
            with tarfile.open(archive_path, "r:") as archive:
                self.assertTrue(archive.getmember("r13229005").isdir())
                self.assertEqual(set(archive.getnames()), {"r13229005", *("r13229005/" + name for name in manifest)})
                for member in archive.getmembers():
                    self.assertEqual((member.mtime, member.uid, member.gid, member.uname, member.gname), (0, 0, 0, "", ""))
                    self.assertEqual(member.mode, 0o755 if member.isdir() else 0o644)
                for name, digest in manifest.items():
                    data = archive.extractfile("r13229005/" + name).read()
                    self.assertEqual(data, (root / name).read_bytes())
                    self.assertEqual(digest, hashlib.sha256(data).hexdigest())
            second_path = archive_path.with_name("second.tar")
            self.assertEqual(make_submission(root, second_path, "r13229005"), manifest)
            self.assertEqual(archive_path.read_bytes(), second_path.read_bytes())

    def test_missing_header_rejected_before_archive_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root, target = Path(temp) / "source", Path(temp) / "submission.tar"
            self.source(root, '#include "absent.hpp"\n')
            with self.assertRaisesRegex(ValueError, "missing submission dependency"):
                make_submission(root, target, "r13229005")
            self.assertFalse(target.exists())

    def test_external_absolute_parent_and_forbidden_dependencies_are_rejected(self):
        headers = ("../outside.hpp", "nested/../outer.hpp", "..\\outside.hpp",
                   "/outside.hpp", "C:/outside.hpp", "C:\\outside.hpp", "\\\\host\\outside.hpp",
                   "tests/test.hpp", "main.cu", "Makefile")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "source"
            self.source(root)
            for header in headers:
                with self.subTest(header=header):
                    (root / "hw3-1.cu").write_text('#include "' + header + '"\n', encoding="utf-8")
                    with self.assertRaises(ValueError):
                        submission_files(root)

    def test_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root, outside = Path(temp) / "source", Path(temp) / "outside.hpp"
            self.source(root, '#include "linked.hpp"\n')
            outside.write_text("outside", encoding="utf-8")
            try:
                (root / "linked.hpp").symlink_to(outside)
            except OSError:
                # Windows may deny symlink creation. Exercise the same resolved
                # containment guard without requiring developer/admin mode.
                original_resolve = Path.resolve
                def resolved(path, strict=False):
                    return outside if path.name == "linked.hpp" else original_resolve(path, strict=strict)
                with patch.object(Path, "resolve", resolved), self.assertRaisesRegex(ValueError, "escapes source root"):
                    submission_files(root)
            else:
                with self.assertRaisesRegex(ValueError, "escapes source root"):
                    submission_files(root)

    def test_invalid_student_ids_extensions_and_existing_archive_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root, target = Path(temp) / "source", Path(temp) / "submission.tar"
            self.source(root)
            for student_id in ("", "../r13229005", "r13229005/", "r13229005 x", "r123", None):
                with self.subTest(student_id=student_id), self.assertRaises(ValueError):
                    make_submission(root, target, student_id)
            with self.assertRaisesRegex(ValueError, "uncompressed"):
                make_submission(root, target.with_suffix(".tgz"), "r13229005")
            self.assertFalse(target.exists())
            target.write_bytes(b"existing evidence")
            with self.assertRaises(FileExistsError):
                make_submission(root, target, "r13229005")
            self.assertEqual(target.read_bytes(), b"existing evidence")


if __name__ == "__main__":
    unittest.main()
