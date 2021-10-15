from setuptools import setup

setup(
    name='puzzlegen',
    url='https://github.com/martius-lab/puzzlegen',
    author='Marco Bagatella',
    author_email='mbagatella@tue.mpg.de',
    packages=['puzzlegen'],
    install_requires=['numpy', 'networkx', 'sklearn', 'gym', 'imageio', 'imageio-ffmpeg', 'opencv-python'],
    version='0.1',
    license='MIT',
    description='Procedurally generated puzzle environments with a gym interface.',
    # long_description=open('README.txt').read(),
)
