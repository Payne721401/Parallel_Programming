#!/usr/bin/env python3
"""Build an exclusive, source-only HW3 submission from quoted dependencies."""
import hashlib
import io
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import tarfile


_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"\r\n]*)"', re.MULTILINE)
_COMMENTS = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\r\n]*|/\*[\s\S]*?\*/')
_STUDENT_ID = re.compile(r'[A-Za-z][0-9]{8}\Z', re.ASCII)
_HEADER_SUFFIXES = {".h", ".hpp", ".cuh"}


def _without_comments(text):
    def replace(match):
        token = match.group()
        return re.sub(r'[^\r\n]', ' ', token) if token.startswith(('//', '/*')) else token
    return _COMMENTS.sub(replace, text.replace('\\\r\n', '').replace('\\\n', ''))


def _include_path(parent, header):
    # Check both syntaxes even when this helper runs on the other operating system.
    portable = header.replace('\\', '/')
    pieces = PurePosixPath(portable)
    if (not header or portable.startswith('/') or PureWindowsPath(header).drive
            or '..' in pieces.parts or '\x00' in header):
        raise ValueError("unsafe quoted include: " + repr(header))
    return parent / Path(*pieces.parts)


def _checked_file(root, relative):
    try:
        resolved = (root / relative).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("missing submission dependency: " + relative.as_posix()) from error
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError("submission dependency escapes source root: " + relative.as_posix())
    if not resolved.is_file():
        raise ValueError("submission dependency is not a file: " + relative.as_posix())
    return resolved


def submission_files(source_root):
    """Return sorted root-relative POSIX filenames, including hw3-1.cu.

    Only quoted includes are followed. The fixed root ca.hpp comes from the TA
    and is deliberately omitted. Other dependencies must be custom headers;
    main.cu, Makefile, tests, and unrelated files never enter the archive.
    """
    root = Path(source_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("source root is not a directory")
    pending = [Path("hw3-1.cu")]
    selected = set()
    inspected = set()
    while pending:
        relative = pending.pop()
        name = relative.as_posix()
        if name in selected:
            continue
        if name == "ca.hpp":
            continue
        if (name != "hw3-1.cu" and relative.suffix not in _HEADER_SUFFIXES
                or "tests" in relative.parts or relative.name == "Makefile"):
            raise ValueError("unsupported submission dependency: " + name)
        source = _checked_file(root, relative)
        selected.add(name)
        # Internal aliases are permitted, but inspect each physical file once.
        # This also bounds dependency cycles through internal directory aliases.
        if source in inspected:
            continue
        inspected.add(source)
        text = _without_comments(source.read_text(encoding="utf-8-sig"))
        for header in _INCLUDE.findall(text):
            dependency = _include_path(relative.parent, header)
            pending.append(dependency)
    return tuple(sorted(selected))


def make_submission(source_root, output_path, student_id):
    """Write an uncompressed .tar once and return relative-name -> SHA256.

    File contents are snapshotted before opening the output. Exclusive creation
    refuses existing files and symlinks; a failed write removes its partial tar.
    """
    if not isinstance(student_id, str) or not _STUDENT_ID.fullmatch(student_id):
        raise ValueError("invalid student ID; expected one letter and eight digits")
    output = Path(output_path)
    if output.suffix != ".tar":
        raise ValueError("submission output must have the uncompressed .tar extension")
    root = Path(source_root).resolve(strict=True)
    contents = {name: _checked_file(root, Path(name)).read_bytes()
                for name in submission_files(root)}
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    # Open before the try: an existing output must never be removed on failure.
    stream = output.open("xb")
    try:
        with stream, tarfile.open(fileobj=stream, mode="w") as archive:
            directory = tarfile.TarInfo(student_id + "/")
            directory.type = tarfile.DIRTYPE
            directory.mode = 0o755
            directory.mtime = directory.uid = directory.gid = 0
            directory.uname = directory.gname = ""
            archive.addfile(directory)
            for name, data in contents.items():
                member = tarfile.TarInfo(student_id + "/" + name)
                member.size = len(data)
                member.mode = 0o644
                member.mtime = member.uid = member.gid = 0
                member.uname = member.gname = ""
                archive.addfile(member, io.BytesIO(data))
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return manifest
