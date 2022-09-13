import numpy as np
import json
from typing import List, Dict, Any, Callable, Union, Optional, TextIO, Tuple
from abc import ABC, abstractmethod, abstractproperty
from scipy.spatial.transform import Rotation as R
from itertools import chain

class HierarchySet(ABC):

    root_segment: Optional['HierarchySet.Segment']
    root_position: List[List]

    @abstractmethod
    def __init__(self, mocap_file, rotation_labels='zyx', position_labels='xyz'):

        self.mocap_file = mocap_file
        self.list_marker = []
        self.list_emg = []
        self.root_segment = None
        self.root_position = [[]]
        self.rotation_labels = rotation_labels
        self.position_labels = position_labels
        self.frame_time = -1
        self.frame_count = 0
        self.hierarchy = {}

    @property
    def depth_first_segment_list(self) -> List['HierarchySet.Segment']:
        def flatten_from_segment(start_segment: 'HierarchySet.Segment'):
            yield start_segment
            for child_segment in start_segment.child_segments:
                flattened_from_child = flatten_from_segment(child_segment)
                for sub_segment in flattened_from_child:
                    yield sub_segment
        flattened_list = list(flatten_from_segment(self.root_segment))
        return flattened_list

    @abstractmethod
    def load_mocap(self):
        raise NotImplementedError()

    def write_to_bvh(self, ofile: TextIO, number_format='.6f'):

        def recursive_segment_write(segment: HierarchySet.Segment, indent_depth: int):

            # region Text parameter extraction
            tabs = '\t' * indent_depth

            node_label = 'ROOT' if segment.is_root else 'JOINT'
            channels = [] if not segment.is_root else \
                ['Xposition' if 'X' in c else 'Yposition' if 'Y' in c else 'Zposition'
                 for c in self.position_labels.upper()]
            channels += ['Xrotation' if 'X' in c else 'Yrotation' if 'Y' in c else 'Zrotation'
                         for c in self.rotation_labels.upper()]
            channel_count = len(channels)
            channel_names = ' '.join(channels)
            offset = ' '.join([('{:' + number_format + '}').format(o) for o in segment.offset])
            zero_offset = ' '.join([('{:' + number_format + '}').format(0) for _ in range(3)])
            # endregion

            # region Write node to file
            ofile.write(f'{tabs}{node_label} {segment.name}\n'
                        f'{tabs}{{\n')

            ofile.write(f'{tabs}\tOFFSET {offset}\n'
                        f'{tabs}\tCHANNELS {channel_count} {channel_names}\n')

            if segment.child_segments:
                for child_segment in segment.child_segments:
                    recursive_segment_write(child_segment, indent_depth+1)
            else:
                ofile.write(f'{tabs}\tEnd Site\n'
                            f'{tabs}\t{{\n'
                            f'{tabs}\t\tOFFSET {zero_offset}\n'
                            f'{tabs}\t}}\n')

            ofile.write(f'{tabs}}}\n')
            # endregion

        def motion_write():
            local_rotations: List[R] = [segment.local_rotation for segment in self.depth_first_segment_list]
            frame_channels = np.hstack([self.root_position, *[rotation.as_euler(self.rotation_labels, True)
                                        for rotation in local_rotations]])
            np.savetxt(ofile, frame_channels, '%'+number_format)

        frame_time_str = ('{:' + number_format + '}').format(self.frame_time)

        ofile.write(f'HIERARCHY\n')
        recursive_segment_write(self.root_segment, 0)
        ofile.write(f'MOTION\n'
                    f'Frames: {self.frame_count}\n'
                    f'Frame Time: {frame_time_str}\n')
        motion_write()

    def save_bvh(self, save_path, number_format='.6f'):
        with open(save_path, 'w') as fp:
            self.write_to_bvh(fp, number_format)

    class Segment:

        rotation: R
        parent_segment: Optional['HierarchySet.Segment']
        child_segments: List['HierarchySet.Segment']
        _rotation_offset: Optional[R]
        _offset: Optional[Tuple]

        def __init__(self, rotation: R, parent=None, rotation_offset: R = None, offset=None, name=None, is_local=False):
            self._rotation_offset = rotation_offset
            self._offset = offset

            self.name = name
            self.parent_segment = parent
            self.child_segments = []
            if self.parent_segment is not None:
                self.parent_segment.add_child(self)

            if not is_local or self.is_root:
                self.rotation = rotation
                self._local_rotation = None
            else:
                self.rotation = rotation * self.parent_segment.rotation
                self._local_rotation = rotation

        def add_child(self, child_segment):
            if child_segment not in self.child_segments:
                self.child_segments.append(child_segment)

        @classmethod
        def from_matrix(cls, rotation_matrix_in_time, normalize=True, *args, **kwargs):
            if normalize:
                rotation_matrix_in_time = HierarchySet.Segment.normalize_axes(rotation_matrix_in_time)
            cls(R.from_matrix(rotation_matrix_in_time))

        @classmethod
        def from_vectors(cls, lateral, frontal, longitudinal, *args, **kwargs):
            _lateral = HierarchySet.Segment.check_input(lateral)
            _frontal = HierarchySet.Segment.check_input(frontal)
            _longitudinal = HierarchySet.Segment.check_input(longitudinal)
            rotation_matrix_in_time = np.array([_lateral, _frontal, _longitudinal])
            cls.from_matrix(rotation_matrix_in_time, *args, **kwargs)

        @property
        def local_rotation(self):
            if self._local_rotation is not None:
                return self.rotation_offset * self._local_rotation
            if self.parent_segment is None:
                return self.rotation_offset * self.rotation
            return self.rotation_offset * self.rotation * self.parent_segment.rotation.inv()

        @property
        def rotation_offset(self):
            if self._rotation_offset is None:
                return R.identity()

        @property
        def offset(self):
            if self._offset is None:
                return 0, 0, 0
            return self._offset

        # region Body segment axes
        @property
        def lateral(self):
            return self.rotation.as_matrix()[:, :, 0]

        @property
        def frontal(self):
            return self.rotation.as_matrix()[:, :, 1]

        @property
        def longitudinal(self):
            return self.rotation.as_matrix()[:, :, 2]

        @property
        def axes(self):
            return [self.lateral, self.frontal, self.longitudinal]
        # endregion

        @property
        def is_root(self):
            return self.parent_segment is None

        @property
        def is_leaf(self):
            return True if self.child_segments else False

        # region Static methods
        @staticmethod
        def check_input(a: np.ndarray):
            try:
                if 3 not in a.shape:
                    raise Exception('Segment only works in 3D')
                if a.ndim > 2:
                    raise Exception('Too many dimensions of input array')
                return np.atleast_2d(a) if a.shape[-1] == 3 else a.T
            except (AttributeError, TypeError):
                raise Exception('Use numpy ndarray-like for Segment objects')

        @staticmethod
        def norm(a):
            out = a / np.linalg.norm(a, axis=1)[:, None]
            return out

        @staticmethod
        def normalize_axes(rotation_matrix_in_time):
            for ax in range(3):
                rotation_matrix_in_time[:, :, ax] = HierarchySet.Segment.norm(rotation_matrix_in_time[:, :, ax])
            return rotation_matrix_in_time
        # endregion

        def __repr__(self):
            return str("Segment: " + self.name)


def main():
    pass


if __name__ == '__main__':
    main()