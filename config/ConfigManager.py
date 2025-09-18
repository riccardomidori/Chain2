import pprint
from configparser import ConfigParser
from pathlib import Path
from config.Utils import auto_str
from config.Logger import XMLLogger


@auto_str
class ConfigManager:
    def __init__(self,
                 utility="MIDORI",
                 config_path=Path("config/config.ini")):
        self.logger = XMLLogger(
            name="ConfigManager",
            path_to_log="log/DigitalManager",
            log_file="server.log",
        ).logger

        self.parsing_jobs = None
        self.parsing_mode = None
        self.parsing = None
        self.generation = None
        self.input_path = None
        self.info_path = None
        self.mysql_config = None
        self.mysql_port = None
        self.mysql_dbname = None
        self.mysql_password = None
        self.mysql_host = None
        self.mysql_user = None
        self.internal_env = None
        self.base_url = None
        self.blob_url = None
        self.container = None
        self.key = None
        self.blob = None
        self.host = None
        self.client_id = None
        self.client_secret = None
        self.input_url = None
        self.username = None
        self.password = None
        self.auth_url = None
        self.port = None
        self.user = None
        self.endpoint_url = None
        self.output_path = None
        self.deactivation_path = None
        self.activation_path = None
        self.consumption_path = None
        self.bucket = None
        self.session_name = None
        self.role = None
        self.secret_key = None
        self.access_key = None
        self.admin_name = None
        self.method = None
        self.env = None
        self.consumption_extension = None
        self.utility = utility
        self.config_path = config_path
        self.config = ConfigParser(interpolation=None)
        self.config.read(config_path)

        self.load()

    def load(self):
        self.env = self.config[self.utility]
        self.internal_env = self.config["ENV"]["env"]
        self.mysql_config = self.config[f"MIDORI-{self.internal_env}"]
        self.mysql_host = self.mysql_config["HOST"]
        self.mysql_port = int(self.mysql_config["PORT"])
        self.mysql_user = self.mysql_config["USER"]
        self.mysql_password = self.mysql_config["PASS"]
        self.mysql_dbname = self.mysql_config["MYDB"]
        self.method = self.env["method"]
        self.admin_name = self.env.get("username")
        self.consumption_extension = self.env.get("consumption_extension")
        self.generation = self.env.get("generation")

        params = self.__dict__
        self.logger.debug(f"Loaded Configuration: {pprint.pformat(params, indent=2)}")


    def load_env(self, env):
        return self.config[env]