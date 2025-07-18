
def get_class_label(filename: str) -> str:
    """
    Extracts the class label from a filename.
    Assumes the filename format is 'classlabel_some_other_info.png'.
    """
    return filename.split("_")[0]