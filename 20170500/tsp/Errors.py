"""
Created on 2020/10/02
@author: nicklee

(Description)
"""


class Error(Exception):
    """
    Base class for exceptions
    """
    pass


class MoveError(Error):
    """
    Exception raised for errors when ants are moving...

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class EmptyClusterError(Error):
    """
    Exception raised for empty clusters.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
