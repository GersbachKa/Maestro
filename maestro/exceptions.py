

class MissingFramesException(Exception):
    def __init__(self, message):
        super().__init__(message)


class UnsupportedFrameFormatException(Exception):
    def __init__(self, message):
        super().__init__(message)


class MultipleFrameFormatException(Exception):
    def __init__(self, message):
        super().__init__(message)


class UnknownFrameTypeException(Exception):
    def __init__(self, message):
        super().__init__(message)


class UnknownFramePathException(Exception):
    def __init__(self, message):
        super().__init__(message)


class UnsupportedMasteringMethodException(Exception):
    def __init__(self, message):
        super().__init__(message)


class MasteringLightsException(Exception):
    def __init__(self, message):
        super().__init__(message)


class UnsetMasteringMethodException(Exception):
    def __init__(self, message):
        super().__init__(message)


class MissingFramesException(Exception):
    def __init__(self, message):
        super().__init__(message)