import argparse
import numpy as np
import io


class RandomizedJoint(object):
    def __init__(self, jtype):
        self.max_length = 2
        self.pi = np.pi
        self.type = jtype
        self.rng = np.random.default_rng()
        self.lower = ''
        self.upper = ''
        if self.type == 'revolute':
            self.upper = f'{self.rng.random() * np.pi}'
            self.lower = f'{-self.rng.random() * np.pi}'
        elif self.type == 'prismatic':
            self.upper = f'{self.rng.random()}'
            self.lower = f'{-self.rng.random()}'
        self.xyz = '{:.2f} {:.2f} {:.2f}'.format(
            np.random.uniform(-self.max_length, self.max_length), np.random.uniform(-self.max_length, self.max_length), np.random.uniform(-self.max_length, self.max_length))
        self.rpy = '{:.2f} {:.2f} {:.2f}'.format(
            np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi))
        self.axis = [0, 1, 0]
        self.rng.shuffle(self.axis)
        self.axis = f'{self.axis}'[1:-1].replace(',', '')

    def to_urdf_str(self, name, parent_id, child_id) -> str:
        r = f'''
  <joint name="joint{name}" type="{self.type}">
    <parent link="link{parent_id}"/>
    <child link="link{child_id}"/>
    <origin xyz="{self.xyz}" rpy="{self.rpy}"/>
    <limit lower="{self.lower}" upper="{self.upper}" effort="0.0" velocity="0.0"/>
    <axis xyz="{self.axis}"/>
  </joint>'''
        return r

    def __str__(self):
        return f'''
Type:  {self.type}
Lower: {self.lower}
Upper: {self.upper}
XYZ:   {self.xyz}
RPY:   {self.rpy}
Axis:  {self.axis}
        '''

    def __repr__(self):
        return self.__str__()


class URDFGenerator(object):
    def __init__(self, **kwargs):
        self.name = kwargs.get('name')
        self.joint_types = ['revolute', 'prismatic', 'continuous', 'fixed']
        self.kwargs = kwargs
        self.joints = []
        self.rng = np.random.default_rng()
        for joint in self.joint_types:
            if kwargs.get(joint):
                self.joints += [RandomizedJoint(joint)
                                for i in range(kwargs.get(joint))]
        self.rng.shuffle(self.joints)
        self.urdf = self._build()

    def _build(self) -> str:
        urdf = io.StringIO()
        urdf.write(f'<robot name="{self.name}">\n')
        for i in range(len(self.joints)+1):
            urdf.write(f'  <link name="link{i}"></link>\n')

        for idx, joint in enumerate(self.joints):
            idxpp = idx + 1
            urdf.write(joint.to_urdf_str(idx, idx, idxpp))
        urdf.write('</robot>')
        urdf.seek(0)
        return urdf.read()

    def __str__(self) -> str:
        print(self.urdf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Randomly generate kinematic chain and create URDF')
    parser.add_argument('name', type=str,
                        help='Number of revolute joints')
    parser.add_argument('--outfile', type=str,
                        help='Filename of the output file. Prints to stdout if not specified')
    parser.add_argument('--revolute', type=int,
                        help='Number of revolute joints')
    parser.add_argument('--prismatic', type=int,
                        help='Number of revolute joints')
    parser.add_argument('--continuous', type=int,
                        help='Number of revolute joints')
    parser.add_argument('--fixed', type=int,
                        help='Number of revolute joints')
    args = parser.parse_args()
    urdf = URDFGenerator(**vars(args))
    if args.outfile:
        with open(args.outfile, 'w') as outfile:
            outfile.write(urdf.urdf)
    else:
        print(urdf.urdf)
