from setuptools import setup

install_requires = [
    "logging",
    "os",
    "urllib",
    "shutil",
    "pathlib",
    "huggingface_hub",
]

setup(
    name='huggingface_sb3',
    version='1.0',
    packages=['huggingface_sb3'],
    url='https://github.com/huggingface/huggingface_sb3',
    license='',
    author='Thomas Simonini',
    author_email='thomas.simonini@huggingface.co',
    description='Additional code for Stable-baselines3 to load and upload models from the Hub.',
    install_requires=install_requires

)
