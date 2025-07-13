import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'dataset_collector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ('share/ament_index/resource_index/packages',
        #     ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # This line finds and installs all launch files
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.py')))
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
            # Assumes your python script is named data_collector.py
            'data_collector = dataset_collector.data_collector:main',
        ],
    },
)