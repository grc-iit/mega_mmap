"""
This module provides classes and methods to launch the MmKmeans application.
MmKmeans is ....
"""
from jarvis_cd.basic.pkg import Application, Color
from jarvis_util import *


class MmHermes(Application):
    """
    This class provides methods to launch the MmKmeans application.
    """
    def _init(self):
        """
        Initialize paths
        """
        pass

    def _configure_menu(self):
        """
        Create a CLI menu for the configurator method.
        For thorough documentation of these parameters, view:
        https://github.com/scs-lab/jarvis-util/wiki/3.-Argument-Parsing

        :return: List(dict)
        """
        return [
            {
                'name': 'nprocs',
                'msg': 'The output path',
                'type': int,
                'default': 16,
            },
            {
                'name': 'ppn',
                'msg': 'The output path',
                'type': int,
                'default': 16,
            },
        ]

    def _configure(self, **kwargs):
        """
        Converts the Jarvis configuration to application-specific configuration.
        E.g., OrangeFS produces an orangefs.xml file.

        :param kwargs: Configuration parameters for this pkg.
        :return: None
        """
        return

    def start(self):
        """
        Launch an application. E.g., OrangeFS will launch the servers, clients,
        and metadata services on all necessary pkgs.

        :return: None
        """
        Exec('mm_hermes_test', MpiExecInfo(env=self.env,
                              nprocs=self.config['nprocs'],
                              ppn=self.config['ppn'],
                              do_dbg=self.config['do_dbg'],
                              dbg_port=self.config['dbg_port']))

    def stop(self):
        """
        Stop a running application. E.g., OrangeFS will terminate the servers,
        clients, and metadata services.

        :return: None
        """
        pass

    def clean(self):
        """
        Destroy all data for an application. E.g., OrangeFS will delete all
        metadata and data directories in addition to the orangefs.xml file.

        :return: None
        """
        pass
