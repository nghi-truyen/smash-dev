import pathlib

if __name__ == "__main__":
    fcore_path = pathlib.Path("../smash/fcore/")

    fcore_wrap_files = sorted(list(fcore_path.glob("*/mwd_*.f90")) + list(fcore_path.glob("*/mw_*.f90")))

    with open("py_mod_names", "w") as f:
        f.write("#% This file is automatically generated by gen_py_mod_names.py" + 2 * "\n")
        f.write("{\n")

        for swf in fcore_wrap_files:
            f.write(f'\t"{swf.stem}": "_{swf.stem}",\n')

        f.write("}")
