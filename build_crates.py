import shutil
import subprocess
import sysconfig
from pathlib import Path

from hatchling.builders.hooks.custom import BuildHookInterface


class BuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        project_root = Path(__file__).parent
        crate_dir = project_root / "src" / "optimizer"
        target_dir = crate_dir / "target" / "release"

        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=crate_dir,
            check=True,
        )

        candidates = sorted(target_dir.glob("liboptimizer*.so"))
        if not candidates:
            candidates = sorted(target_dir.glob("optimizer*.pyd"))
        if not candidates:
            candidates = sorted(target_dir.glob("optimizer*.dll"))
        if not candidates:
            raise RuntimeError("No optimizer extension module found after cargo build.")

        ext_path = candidates[0]
        dest_dir = project_root / "src" / "greentorch"
        dest_dir.mkdir(parents=True, exist_ok=True)
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        dest_path = dest_dir / f"optimizer{ext_suffix}"
        shutil.copy2(ext_path, dest_path)

        force_include = build_data.setdefault("force_include", {})
        force_include[str(dest_path)] = f"greentorch/{dest_path.name}"