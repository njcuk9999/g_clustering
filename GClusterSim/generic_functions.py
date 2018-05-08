#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2018-05-08 at 11:25

@author: cook
"""

import time
import sys

# =============================================================================
# Define variables
# =============================================================================

# -----------------------------------------------------------------------------

# =============================================================================
# Define functions
# =============================================================================
# defines the colours
class BColors:
    HEADER = '\033[95;1m'
    OKBLUE = '\033[94;1m'
    OKGREEN = '\033[92;1m'
    WARNING = '\033[93;1m'
    FAIL = '\033[91;1m'
    ENDC = '\033[0;0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def printlog(message, level=''):

    pcolour = printcolour(level)
    message = message.split('\n')
    for mess in message:
        unix_time = time.time()
        human_time = time.strftime('%H:%M:%S', time.localtime(unix_time))
        dsec = int((unix_time - int(unix_time)) * 100)
        pargs = [pcolour, human_time, dsec, mess, BColors.ENDC]
        print('{0}{1}.{2:02d} | {3}{4}'.format(*pargs))


def printcolour(level=''):
    if level == 'error':
        return BColors.FAIL
    elif level == 'warning':
        return BColors.WARNING
    elif level == 'info':
        return BColors.OKBLUE
    elif level is None:
        return BColors.ENDC
    else:
        return BColors.OKGREEN


def exit(level):
    sys.exit(1)

# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # print 'Hello World!'
    print("Hello World!")

# =============================================================================
# End of code
# =============================================================================
