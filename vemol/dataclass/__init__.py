from .config import BaseDataclass
from .config import CommonConfig
from .config import Config
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
node = CommonConfig()
node._name = 'base_common'
cs.store(name='base_common', group="common", node=node)

node = Config()
node._name = 'base_config'
cs.store(name='base_config', node=node)