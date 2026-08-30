from pathlib import Path

import pytest

from apple_bench.data.appleleaf9 import APPLELEAF9_TO_PV, AppleLeaf9


def test_label_mapping():
    """Class dir names use the actual capitalization from the AppleLeaf9 GitHub repo."""
    assert APPLELEAF9_TO_PV["Health"] == "Apple___healthy"
    assert APPLELEAF9_TO_PV["Scab"] == "Apple___Apple_scab"
    assert APPLELEAF9_TO_PV["Rust"] == "Apple___Cedar_apple_rust"
    # Unmapped classes must not appear:
    assert "Alternaria leaf spot" not in APPLELEAF9_TO_PV
    assert "Mosaic" not in APPLELEAF9_TO_PV


def test_only_mapped_classes_loaded(tmp_path: Path):
    for cls in ["Health", "Scab", "Rust", "Mosaic"]:
        (tmp_path / cls).mkdir()
        (tmp_path / cls / "0.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    ds = AppleLeaf9(root=tmp_path, transform=None)
    assert len(ds) == 3
    assert set(ds.targets) == {0, 2, 3}  # scab, cedar_rust, healthy


def test_subdirectories_are_skipped(tmp_path: Path):
    """Accidental subdirectories must not be indexed (matches PV adapter test).

    Note: is_file() does not exclude dot-files like .DS_Store; that is a
    known minor gap shared with the PV adapter and is intentional — keeping
    a conservative is_file() filter avoids hard-coded extension allowlists
    that could silently drop valid imagery (e.g., .tif, .webp).
    """
    (tmp_path / "Health").mkdir()
    (tmp_path / "Health" / "0.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (tmp_path / "Health" / "1.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (tmp_path / "Health" / "subdir").mkdir()
    (tmp_path / "Health" / "subdir" / "x.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    ds = AppleLeaf9(root=tmp_path, transform=None)
    # 2 image files; subdir excluded by is_file().
    assert len(ds) == 2


def test_root_with_no_mapped_classes_raises(tmp_path: Path):
    """If none of the mapped classes are present, raise — prevents silently empty dataset."""
    (tmp_path / "Alternaria leaf spot").mkdir()
    (tmp_path / "Alternaria leaf spot" / "0.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    with pytest.raises(FileNotFoundError, match="No mapped AppleLeaf9 class dirs"):
        AppleLeaf9(root=tmp_path, transform=None)
