from markersets.convertsets.hierarchyset import HierarchySet
from bvh import Bvh, BvhNode
import numpy as np
from scipy.spatial.transform import Rotation as R

BVH_TRANSLATION_CHANNELS = "Xposition", "Yposition", "Zposition"
BVH_ROTATION_CHANNELS = "Zrotation", "Yrotation", "Xrotation"


class BvhSet(HierarchySet):
    def __init__(self, mocap_file, rotation_labels='zyx', position_labels='xyz'):
        super(BvhSet, self).__init__(mocap_file=mocap_file,
                                            rotation_labels=rotation_labels, position_labels=position_labels)

    def load_mocap(self):
        with open(self.mocap_file) as f:
            mocap = Bvh(f.read())
        parent_child_names = list(BvhSet.depth_first_node_name_search(mocap.root))

        frames = np.array(mocap.frames, dtype=float)

        def get_frames(j_in, chs_in):
            joint_index = mocap.get_joint_channels_index(j_in)
            channel_indices = np.array([mocap.get_joint_channel_index(j_in, ch) for ch in chs_in])
            return frames[:, joint_index + channel_indices]

        root_name = parent_child_names[0][0]
        self.root_position = get_frames(root_name, BVH_TRANSLATION_CHANNELS)

        root_rotation = R.from_euler(self.rotation_labels, get_frames(root_name, BVH_ROTATION_CHANNELS), degrees=True)
        self.root_segment = HierarchySet.Segment(root_rotation, name=root_name, offset=mocap.joint_offset(root_name))
        self.hierarchy[root_name] = self.root_segment

        for parent_name, child_name in parent_child_names:
            rotation = R.from_euler(self.rotation_labels, get_frames(child_name, BVH_ROTATION_CHANNELS), degrees=True)
            offset = mocap.joint_offset(child_name)
            segment = HierarchySet.Segment(rotation, self.hierarchy[parent_name],
                                           offset=offset, name=child_name, is_local=True)
            self.hierarchy[child_name] = segment

        self.frame_count = mocap.nframes
        self.frame_time = mocap.frame_time

    def get_emg_data(self, file_reader=None):
        pass

    def save_json(self, save_path, indent=None, **kwargs):
        raise NotImplementedError()

    @property
    def hip(self):
        return None

    @staticmethod
    def depth_first_node_name_search(root_node: BvhNode):

        if len(root_node.value) > 0 and root_node.value[0] == "JOINT":
            yield root_node.parent.name, root_node.name
        for child in root_node:
            recursive_call = BvhSet.depth_first_node_name_search(child)
            for segment_name in recursive_call:
                yield segment_name


def main():
    path = r'tests/test_freebvh.bvh'
    mocap = BvhSet(path)
    mocap.se_load_mocap()
    mocap.save_bvh("test.bvh")
    pass


if __name__ == '__main__':
    main()