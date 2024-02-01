import pathlib as pl

# some paths to important directories
script_path = pl.Path(__file__).resolve()
utils_dir_path = script_path.parent
src_dir_path = utils_dir_path.parent
project_root_path = src_dir_path.parent
resources_dir_path = project_root_path.joinpath("resources")