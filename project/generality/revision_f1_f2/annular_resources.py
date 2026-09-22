"""512-source Annular geometry and bounded equilibrium-search protocol."""
import numpy as np
from compute_appendix_case import finite_ka_protocol as original_protocol

RING_COUNTS = (32, 38, 45, 51, 58, 64, 70, 76, 78)
DESCRIPTION = ('Nine concentric planar rings, radii 52.0 to 133.6 mm in 10.2 mm steps; '
    '32, 38, 45, 51, 58, 64, 70, 76 and 78 sources, respectively; normals +z. '
    'The 52 mm inner opening and minimum 10 mm physical spacing are retained.')

def geometry():
    rings=[]
    for ring,count in enumerate(RING_COUNTS):
        radius=.052+ring*.0102
        angle=(np.arange(count)+.5*(ring%2))*2*np.pi/count
        rings.append(np.c_[radius*np.cos(angle),radius*np.sin(angle),np.zeros(count)])
    positions=np.vstack(rings)
    normals=np.tile([0.,0.,1.],(len(positions),1))
    assert positions.shape==(512,3)
    return positions,normals

def validation_protocol(mode='smoke',setting_id='A07'):
    result=original_protocol('smoke',setting_id)
    result.update(max_nfev_per_start=100,
        budget_note='Same force model, seven starts, search domain and root tolerance; increased local evaluations from 22 to 100.')
    return result
