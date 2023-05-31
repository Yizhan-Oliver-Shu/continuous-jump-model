from regime.simulation_helper import *

import imgkit


def add_cluster(string, cluster):
    return string if not cluster else f"{string}-cluster"

def save_df_as_fig(df, fig_path):
    check_dir_exist(fig_path)
    return imgkit.from_string(df.style.to_html(), fig_path)
    
def summary_into_figures(path, key_data, cluster=False):
    if isinstance(key_data, list):
        for key_data_ in key_data: summary_into_figures(path, key_data_, cluster)
        return 
    summary_str, figure_str = add_cluster("summary", cluster), add_cluster("figure", cluster)
    folder = f"{path[summary_str]}/{key_data}"
    filnames = filter_filenames_in_folder(folder, "summary")
    for filename in filnames:
        fig_name = f"{path[figure_str]}/{key_data}/{filename}".replace("h5", "jpeg")
        save_df_as_fig(pd.read_hdf(f"{folder}/{filename}"), fig_name)