from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

ext = Extension("laplace_cython",
	sources=["laplace_cython.pyx"],
    include_dirs = [numpy.get_include()],
	extra_compile_args=['-O3'],
	language="c"
	)
                
setup(
	  name = "laplace_cython",
	  ext_modules=[ext],
      cmdclass = {'build_ext': build_ext})