"""
This module provides classes and methods to launch the MmGadget2conv application.
MmGadget2conv is ....
"""
from jarvis_cd.basic.pkg import Application
from jarvis_util import *


class MmGadget2conv(Application):
    """
    This class provides methods to launch the MmGadget2conv application.
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
                'name': 'in',
                'msg': 'The input path of the HDF5 file output by gadget2',
                'type': str,
                'default': None,
            },
            {
                'name': 'out',
                'msg': 'The output path of the binary file to create',
                'type': str,
                'default': None,
            },
            {
                'name': 'nprocs',
                'msg': 'Number of procs',
                'type': int,
                'default': 16,
            },
            {
                'name': 'ppn',
                'msg': 'Processes per node',
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
        out = self.config['out']
        Mkdir(str(pathlib.Path(out).parent),
              LocalExecInfo(env=self.env))

    def start(self):
        """
        Launch an application. E.g., OrangeFS will launch the servers, clients,
        and metadata services on all necessary pkgs.

        :return: None
        """
        cmd = [
            'mm_gadget2conv',
            self.config['in'],
            self.config['out']
        ]
        cmd = ' '.join(cmd)
        Exec(cmd, MpiExecInfo(nprocs=self.config['nprocs'],
                              ppn=self.config['ppn'],
                              do_dbg=self.config['do_dbg'],
                              dbg_port=self.config['dbg_port'],
                              env=self.mod_env))

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
