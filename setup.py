from setuptools import setup

install_requires = [
    "huggingface_hub~=0.8",
    "pyyaml~=6.0",
    "wasabi",
    "numpy",
    "cloudpickle>=1.6"
]

extras = {}

extras["quality"] = [
    "black~=22.0",
    "isort>=5.5.4",
    "flake8>=3.8.3",
]

setup(
    name='huggingface_sb3',
    version='2.2.4',
    packages=['huggingface_sb3'],
    url='https://github.com/huggingface/huggingface_sb3',
    license='Apache',
    author='Thomas Simonini, Omar Sanseviero and Hugging Face Team',
    author_email='thomas.simonini@huggingface.co',
    description='Additional code for Stable-baselines3 to load and upload models from the Hub.',
    install_requires=install_requires,
    extras_require=extras,
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="reinforcement learning deep reinforcement learning RL",
)
