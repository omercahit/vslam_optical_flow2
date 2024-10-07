from setuptools import find_packages, setup

package_name = 'vslam_optical_flow2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools', 'opencv-python'],
    zip_safe=True,
    maintainer='omercahit',
    maintainer_email='o.cahitozdemir@gmail.com',
    description='vslam_optical_flow for ros2',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vslam_optical_flow2 = vslam_optical_flow2.vslam_optical_flow2:main',
        ],
    },
)
