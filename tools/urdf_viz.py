import inspect
import argparse
import sys

from dlkinematics.urdf import chain_from_urdf_file 
from dlkinematics.urdf_parser.urdf import Robot


class URDFVisualiser(object):
    def __init__(self, chain: Robot):
        self.chain = chain
        self.root = self.chain.get_root()
        self.joint_types = ['revolute', 'prismatic', 'continuous', 'fixed']

    def print_chain_recursive(self):
        self._print_chain_recursive(self.root)

    def print_info(self):
        print(f'Name: \t\t {self.chain.name}')
        print(f'Total Joints: \t {len(self.chain.joints)}')
        print(f'Total Links: \t {len(self.chain.links)}')
        for jtype in self.joint_types:
            print(f'# {jtype}: \t {self._num_type_joints(jtype)}')
        print()

    def _num_type_joints(self, type):
        return len(list(filter(lambda x: x.type == type, self.chain.joints)))

    def _print_chain_recursive(self, next_link, depth=0):
        print('\t' * depth, '⚙️ ', next_link)
        depth += 1
        children = self.chain.child_map.get(next_link)
        if children != None:
            for child in self.chain.child_map.get(next_link):
                print('\t' * depth, f'{self.chain.joint_map.get(child[0]).type}_{self._rotation_axis(self.chain.joint_map.get(child[0]).axis)}')
                self._print_chain_recursive(child[1], depth)

    @staticmethod
    def is_valid_urdf(parser, file):
        try:
            chain_from_urdf_file(file)
        except Exception as e:
            return parser.error(e)

        return chain_from_urdf_file(file)
    
    def _rotation_axis(self, axis):
        res = ''
        try:
            axis = [abs(a) for a in axis]
            res = ['x', 'y', 'z'][axis.index(1)]
        except:
            pass
        return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualise and analyse your URDF file')
    parser.add_argument('urdf_file', type=lambda f: URDFVisualiser.is_valid_urdf(parser, f),
                        help='URDF file to be visualised and analysed')
    args = parser.parse_args()

    vis = URDFVisualiser(args.urdf_file)
    # print(vis.chain.child_map)
    vis.print_info()
    vis.print_chain_recursive()
