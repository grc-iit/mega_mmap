"""
This module provides classes and methods to launch the MmRandomForest application.
MmRandomForest is ....
"""
from jarvis_cd.basic.pkg import Application, Color
from jarvis_util import *


class MmRandomForest(Application):
    """
    This class provides methods to launch the MmRandomForest application.
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
                'name': 'train_path',
                'msg': 'The input data path',
                'type': str,
                'default': None,
            },
            {
                'name': 'test_path',
                'msg': 'The input data path',
                'type': str,
                'default': None,
            },
            {
                'name': 'window_size',
                'msg': 'The max amount of memory to consume',
                'type': str,
                'default': '1g',
            },
            {
                'name': 'nprocs',
                'msg': 'The output path',
                'type': int,
                'default': 1,
            },
            {
                'name': 'ppn',
                'msg': 'The output path',
                'type': int,
                'default': 16,
            },
            {
                'name': 'api',
                'msg': 'The implementation to use',
                'type': str,
                'default': 'spark',
                'choices': ['spark', 'mega', 'pandas']
            },
            {
                'name': 'nfeature',
                'msg': 'The number of features for prediction',
                'type': str,
                'default': '2'
            },
            {
                'name': 'scratch',
                'msg': 'Where spark buffers data',
                'type': str,
                'default': '${HOME}/sparktmp/',
            },
            {
                'name': 'num_trees',
                'msg': 'The number of trees to generate',
                'type': int,
                'default': 1
            },
            {
                'name': 'max_depth',
                'msg': 'Max depth of trees',
                'type': int,
                'default': 4
            },
        ]

    def _configure(self, **kwargs):
        """
        Converts the Jarvis configuration to application-specific configuration.
        E.g., OrangeFS produces an orangefs.xml file.

        :param kwargs: Configuration parameters for this pkg.
        :return: None
        """
        self.config['scratch'] = os.path.expandvars(self.config['scratch'])
        Mkdir(self.config['scratch'],
              PsshExecInfo(hosts=self.jarvis.hostfile))

    def start(self):
        """
        Launch an application. E.g., OrangeFS will launch the servers, clients,
        and metadata services on all necessary pkgs.

        :return: None
        """
        mm_kmeans = ['mmap', 'mega']
        if self.config['api'] == 'spark':
            master_host = self.env['SPARK_MASTER_HOST']
            master_port = self.env['SPARK_MASTER_PORT']
            cmd = [
                f'{self.env["MM_PATH"]}/scripts/spark_random_forest.py',
                self.config['train_path'],
                self.config['test_path'],
                str(self.config['num_trees']),
                str(self.config['max_depth']),
            ]
            cmd = ' '.join(cmd)
            SparkExec(cmd, master_host, master_port,
                      exec_info=LocalExecInfo(env=self.env))
        elif self.config['api'] == 'pandas':
            cmd = [
                'python3',
                f'{self.env["MM_PATH"]}/scripts/pandas_random_forest.py',
                self.config['train_path'],
                self.config['test_path'],
                str(self.config['num_trees']),
                str(self.config['max_depth']),
            ]
            cmd = ' '.join(cmd)
            Exec(cmd, LocalExecInfo(env=self.env))
        elif self.config['api'] in mm_kmeans:
            cmd = [
                'mm_random_forest',
                self.config['api'],
                self.config['train_path'],
                self.config['test_path'],
                self.config['nfeature'],
                self.config['window_size'],
            ]
            cmd = ' '.join(cmd)
            Exec(cmd, MpiExecInfo(env=self.env,
                                  nprocs=self.config['nprocs'],
                                  ppn=self.config['ppn'],
                                  do_dbg=self.config['do_dbg'],
                                  dbg_port=self.config['dbg_port']))

    def _get_stat(self, stat_dict):
        """
        Parse the output of the application to extract performance statistics.

        :param stat_dict: A dictionary to store the performance statistics.
        :return: None
        """
        parser = MonitorParser(self.env['MONITOR_DIR'])
        parser.parse()
        stat_dict[f'{self.pkg_id}.runtime'] = self.start_time
        stat_dict[f'{self.pkg_id}.avg_mem'] = parser.avg_memory()
        stat_dict[f'{self.pkg_id}.peak_mem'] = parser.peak_memory()
        stat_dict[f'{self.pkg_id}.avg_cpu'] = parser.avg_cpu()

    def stop(self):
        """
        Stop a running application. E.g., OrangeFS will terminate the servers,
        clients, and metadata services.

        :return: None
        """
        pass

    def kill(self):
        """
        Kill a running application. E.g., OrangeFS will terminate the servers,
        clients, and metadata services.

        :return: None
        """
        Kill('.*mm_random_forest.*', PsshExecInfo(hosts=self.jarvis.hostfile))

    def clean(self):
        """
        Destroy all data for an application. E.g., OrangeFS will delete all
        metadata and data directories in addition to the orangefs.xml file.

        :return: None
        """
        pass
