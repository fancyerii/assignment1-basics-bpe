from setuptools import setup, Extension
from Cython.Build import cythonize
import os
import sysconfig
 
include_path = sysconfig.get_path('include')
internal_include_path = os.path.join(include_path, 'internal')

print(f"{internal_include_path=}")
project_root = os.path.dirname(os.path.abspath(__file__))

maxheapqc_module = Extension('cs336_basics.maxheapqc', sources=['cs336_basics/_maxheapqmodule.c'],
        extra_compile_args=[
            f'-I{internal_include_path}',
        ]
        )

setup(
    name='maxheapqc',
    version='1.0',
    description='maxheapqc',
    packages=['cs336_basics'],
    ext_modules=[maxheapqc_module]
)

setup(
    packages=['cs336_basics'],
    name='cs336_basics.bpe_updater',
    ext_modules=cythonize("cs336_basics/bpe_updater.pyx"),

)


ext_modules = [
    Extension(
        name="cs336_basics.bpe_train_step2_wrapper",
        sources=["cs336_basics/bpe_train_step2_wrapper.pyx"],

        language="c++",
        #extra_compile_args=['-std=c++17', '-O3'],
        extra_compile_args=['-std=c++17'],
        libraries=["bpe_train_step2"],

        library_dirs=[f"{project_root}/lib_bpe_train_step2/lib"],
        runtime_library_dirs=[f"{project_root}/lib_bpe_train_step2/lib"],
        include_dirs=[f"{project_root}/lib_bpe_train_step2/include",
                      f"{project_root}/lib_bpe_train_step2/include/emhash"],
    )
]

setup(
    packages=['cs336_basics'],
    name='bpe_train_step2',
    ext_modules=cythonize(ext_modules),

)
