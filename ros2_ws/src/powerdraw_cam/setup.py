from setuptools import setup

package_name = 'powerdraw_cam'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='antony',
    maintainer_email='antony@todo.com',
    description='Camera publisher for V4L2 loopback device',
    license='MIT',
    entry_points={
        'console_scripts': [
            'camera_pub = powerdraw_cam.camera_pub:main',
        ],
    },
)
