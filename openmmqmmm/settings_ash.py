"""Transitional in-memory settings.

The ~/ash_user_settings.ini machinery has been removed: ORCA discovery now
uses the orcadir argument / OPENMMQMMM_ORCADIR environment variable / PATH
(see interfaces.interface_ORCA.find_orca), and the connectivity defaults live
in modules.module_coords. The two flags below only feed the legacy print
machinery and disappear with it when output moves to the logging module.
"""

settings_dict = {
    # Whether to use ANSI color escape sequences in printed output
    "use_ANSI_color": False,
    # Extra debug printing (printdebug)
    "debugflag": False,
}
