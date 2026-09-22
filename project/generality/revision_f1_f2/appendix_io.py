"""Atomic durable writes for figure and archive transfer between work sessions."""
from pathlib import Path
from io import BytesIO
import os
import tempfile

def write_bytes(path, data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            # Large writes on the workspace filesystem can be short. Advance
            # by the bytes actually written and keep each request bounded.
            view=memoryview(data)
            offset=0
            while offset < len(view):
                written=stream.write(view[offset:offset+8*1024*1024])
                if not written:
                    raise OSError('No progress while writing '+str(path))
                offset+=written
            stream.flush()
            os.fsync(stream.fileno())
            if stream.tell()!=len(view):
                raise OSError('Incomplete write for '+str(path))
        os.replace(temporary,path)
        return path
    finally:
        if os.path.exists(temporary): os.unlink(temporary)

def save_figure(figure, path, **kwargs):
    path=Path(path)
    stream=BytesIO()
    figure.savefig(stream,format=path.suffix[1:],**kwargs)
    value=stream.getvalue()
    if path.suffix=='.png':
        from PIL import Image
        with Image.open(BytesIO(value)) as im: im.verify()
    return write_bytes(path,value)

def save_npz(path, **arrays):
    import numpy as np
    stream=BytesIO()
    np.savez_compressed(stream,**arrays)
    return write_bytes(path,stream.getvalue())
