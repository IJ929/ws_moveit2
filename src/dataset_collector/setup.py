import os
from glob import glob
from setuptools import setup

package_name = 'dataset_collector'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Include all config files
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ij',
    maintainer_email='sandy10203@gmail.com',
    description='A ROS 2 package to collect a dataset for Franka Emika Panda.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # This makes 'data_collector.py' available as an executable
            'data_collector = dataset_collector.data_collector:main',
        ],
    },
)