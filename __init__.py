""" Package wide definitions.
Package-wide definitions for logging style, exit function, and configuration variable.

This module sets up colored and formatted logging output for different log levels,
including a custom 'NOTIMPLEMENTED' level. It also initializes a logger for the package
and logs the loading of configuration.

Attributes:
    BOLD (str): ANSI escape code for bold text.
    RESET (str): ANSI escape code to reset text formatting.
    logger (logging.Logger): Logger instance for the package.

Logging Levels:
    INFO: Magenta
    DEBUG: Cyan
    WARNING: Yellow
    ERROR: Red
    NOTIMPLEMENTED: Green (custom level)

Example:

Logging Style
exit() function
configuration variable


"""

#pretty logging
import logging

logging.addLevelName( logging.INFO,    "\033[1;95m{0}\033[1;0m".format('INFO '))
logging.addLevelName( logging.DEBUG,   "\033[1;96m{0}\033[1;0m".format('DEBUG'))
logging.addLevelName( logging.WARNING, "\033[1;93m{0}\033[1;0m".format('WARN '))
logging.addLevelName( logging.ERROR,   "\033[1;91m{0}\033[1;0m".format('ERROR'))

#add user defined level
logging.addLevelName( 9 ,              "\033[1;92m%{0}\033[1;0m".format('NOTIMPLEMENTED'))

BOLD = "\033[1m"
RESET = "\033[0m"
logging.basicConfig( level=logging.INFO, format='{0}[%(asctime)s.%(msecs)03d]{1} %(levelname)8s  {0}%(name)-50s{1}%(message)s'.format(BOLD,RESET),datefmt="%H:%M:%S")

logger = logging.getLogger(__name__)
logger.info("Loading configuration.")