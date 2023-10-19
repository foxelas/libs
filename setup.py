#!/usr/bin/env python
import glob
import os
import shutil
from distutils.core import setup
from distutils.cmd import Command

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    CLEAN_FILES = './build ./dist ./*.pyc ./*.tgz ./*.egg-info'.split(' ')

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        global here

        here = os.getcwd()
        for path_spec in self.CLEAN_FILES:
            # Make paths absolute and relative to this path
            abs_paths = glob.glob(os.path.normpath(os.path.join(here, path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(here):
                    # Die if path in CLEAN_FILES is absolute + outside this directory
                    raise ValueError("%s is not a path inside %s" % (path, here))
                print(f"Cleaning {os.path.relpath(path)}")
                shutil.rmtree(path)

setup(
    name='libs',
    version='1.0.0',
    packages=['libs', 'libs.pytube', 'libs.pytube.contrib', 'libs.streams', 'libs.anomalib', 'libs.anomalib.data',
              'libs.anomalib.data.base', 'libs.anomalib.data.utils', 'libs.anomalib.data.utils.generators',
              'libs.anomalib.utils', 'libs.anomalib.utils.cv', 'libs.anomalib.utils.cli', 'libs.anomalib.utils.hpo',
              'libs.anomalib.utils.sweep', 'libs.anomalib.utils.sweep.helpers', 'libs.anomalib.utils.loggers',
              'libs.anomalib.utils.metrics', 'libs.anomalib.utils.callbacks', 'libs.anomalib.utils.callbacks.nncf',
              'libs.anomalib.utils.callbacks.visualizer', 'libs.anomalib.config', 'libs.anomalib.deploy',
              'libs.anomalib.deploy.inferencers', 'libs.anomalib.models', 'libs.anomalib.models.cfa',
              'libs.anomalib.models.dfm', 'libs.anomalib.models.rkde', 'libs.anomalib.models.cflow',
              'libs.anomalib.models.dfkde', 'libs.anomalib.models.draem', 'libs.anomalib.models.padim',
              'libs.anomalib.models.stfpm', 'libs.anomalib.models.ai_vad', 'libs.anomalib.models.ai_vad.clip',
              'libs.anomalib.models.csflow', 'libs.anomalib.models.fastflow', 'libs.anomalib.models.ganomaly',
              'libs.anomalib.models.patchcore', 'libs.anomalib.models.components',
              'libs.anomalib.models.components.base', 'libs.anomalib.models.components.flow',
              'libs.anomalib.models.components.stats', 'libs.anomalib.models.components.layers',
              'libs.anomalib.models.components.filters', 'libs.anomalib.models.components.sampling',
              'libs.anomalib.models.components.classification', 'libs.anomalib.models.components.feature_extractors',
              'libs.anomalib.models.components.dimensionality_reduction', 'libs.anomalib.models.efficientad',
              'libs.anomalib.models.reverse_distillation', 'libs.anomalib.models.reverse_distillation.components',
              'libs.anomalib.pre_processing', 'libs.anomalib.pre_processing.transforms',
              'libs.anomalib.post_processing', 'libs.anomalib.post_processing.normalization', 'libs.foxutils',
              'libs.foxutils.utils', 'libs.foxutils.utils.lightning_models', 'libs.foxutils.gradio',
              'libs.foxutils.streams', 'libs.foxutils.feature_extractors'],
    classifiers=[
        'Development Status :: 0 - Dev',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    cmdclass={
        'clean': CleanCommand,
    },
    url='https://github.com/foxelas?tab=repositories',
    license='MIT',
    author='github:foxelas',
    author_email='foxelas@outlook.com',
    description='',
    project_urls={
        'Documentation': '',
        'Source': '',
    },
)

