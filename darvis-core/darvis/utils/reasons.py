from enum import Enum


class TransitionReasons(Enum):
    USER_CANCEL = "User canceled action."
    USER_QUIT = "User requested shutdown."
    INVALID_EVENT = "No transition setup for this event."
