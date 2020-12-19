import base64
import os

import IPython.display

def show_video_in_notebook(video_path, width=500, height='auto', autoplay=True,
                           embed=True):
    """
    Helper function to show a video in a jupyter notebook.
    :param video_path: Path to video file.
    :param width: Width of video element on the page.
    :param height: Height of video element on the page.
    :param autoplay: Whether video should automatically start playing.
    :param embed: Whether to embed the video in the notebook itself,
    or just link to it. Linking won't work if file is outside servers pwd;
    embedding won't work if video is too large.
    :return: An IPython HTML object that jupyter notebook can display.
    """
    video_path = os.path.relpath(video_path, start=os.path.curdir)
    autoplay = 'autoplay' if autoplay else ''

    if embed:
        with open(video_path, "rb") as f:
            encoded = base64.b64encode(f.read(), ).decode("ascii")
        _, ext = os.path.splitext(video_path)
        src_str = f"data:video/{ext[1:]};base64,{encoded}"
    else:
        src_str = f"{video_path}"

    raw_html = f'<video src="{src_str}" controls {autoplay} ' \
               f'width="{width}" height="{height}" />'

    return IPython.display.HTML(data=raw_html)
