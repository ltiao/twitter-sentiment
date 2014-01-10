import os
import logging.config

import yaml

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '%(asctime)s [%(levelname)s] (%(threadName)-10s): %(message)s',
            'datefmt': '%m/%d/%Y %I:%M:%S %p'
        },
        'simple': {
            'format': '%(asctime)s %(levelname)s %(message)s',
            'datefmt': '%m/%d/%Y %I:%M:%S %p'
        },
    },
    'handlers': {
        'default': {
            'level':'INFO',
            'class':'logging.handlers.RotatingFileHandler',
            'filename': 'twitter.log',
            'maxBytes': 1024 * 1024, # 1 mb,
            'backupCount': 15,
            'formatter': 'simple',
        },
        'console':{
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': True,
        },
        'twitter': {
            'handlers': ['default', 'console'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'eval': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False,
        },
    }
}

def setup_logging(default_path='log_settings.yml', default_level=logging.INFO, env_key='LOG_CFG'):
    """Setup logging configuration

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)