from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

MAAI_VERSION = "0.2.18"
MAAI_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/68/08/"
    "6135aae79652c212bc16ed582c219745001a1d0449ad6dbc796687ee08a6/"
    "maai-0.2.18-py3-none-any.whl"
)
MAAI_WHEEL_SHA256 = "4402f02d520de0129cba72ebfa52a9a184ba6e5e353719cc1d4ab9fcc8887f29"
CPC_REVISION = "4dacbba1ca1cc54861e1adce390594f8f8450ffe"
CPC_URL = (
    f"https://raw.githubusercontent.com/inokoj/VAP-Realtime/{CPC_REVISION}/"
    "asset/cpc/60k_epoch4-d0f474de.pt"
)
CPC_SHA256 = "d0f474de968c7243c20f43dee1626681618f2f150d3e638e82b79e22563b17f1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(destination) != expected_sha256:
            raise ValueError(f"checksum mismatch for existing asset: {destination}")
        return
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}-", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        urllib.request.urlretrieve(url, temporary)
        actual = _sha256(temporary)
        if actual != expected_sha256:
            raise ValueError(
                f"download checksum mismatch for {destination.name}: "
                f"expected {expected_sha256}, got {actual}"
            )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parent
    vendor = root / "vendor"
    maai_init = vendor / "maai" / "__init__.py"
    if not maai_init.exists():
        wheel = root / "assets" / f"maai-{MAAI_VERSION}.whl"
        _download(MAAI_WHEEL_URL, wheel, MAAI_WHEEL_SHA256)
        vendor.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(vendor)

    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        raise ValueError("HF_HOME must be set before preparing Realtime-VAP")
    cpc = Path(hf_home) / "turn-eval-assets" / "realtime-vap" / "60k_epoch4-d0f474de.pt"
    _download(CPC_URL, cpc, CPC_SHA256)
    return vendor, cpc


if __name__ == "__main__":
    vendor_path, cpc_path = prepare()
    print(f"maai_vendor={vendor_path}")
    print(f"cpc_model={cpc_path}")
