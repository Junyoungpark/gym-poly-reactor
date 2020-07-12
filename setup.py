from setuptools import setup, find_packages

setup(
    name='gym_poly_reactor',
    version='0.0',
    description='OpenAI gym style industrial polymerization reactor simulator',
    author='Junyoung Park',
    author_email='junyoungpark@kaist.ac.kr',
    url='https://github.com/Junyoungpark/gym-poly-reactor',
    install_requires=['numpy', 'gym', 'do_mpc'],
    packages=find_packages(exclude=['docs', 'tests*']),
    keywords=['gym_poly_reactor'],
    python_requires='>=3.6',
    zip_safe=False
)
